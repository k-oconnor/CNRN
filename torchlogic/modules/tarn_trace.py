import logging
from typing import Optional

import numpy as np
import torch
from torch import nn

from torchlogic.utils.explanations import register_hooks, simplification
from torchlogic.sklogic.base.base_estimator import BaseSKLogicEstimator

from ..nn import LukasiewiczChannelOrBlock, LukasiewiczChannelAndBlock, Predicates


class TARNRNTraceModule(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            layer_sizes: list,
            feature_names: list,
            n_selected_features_input: int,
            n_selected_features_internal: int,
            n_selected_features_output: int,
            perform_prune_quantile: float,
            ucb_scale: float,
            head_layer_sizes: Optional[list] = None,
            mlp_head_hidden_dim: int = 0,
            normal_form: str = 'dnf',
            add_negations: bool = False,
            weight_init: float = 0.2,
            logits: bool = False
    ):
        """
        Initialize a Bandit Reinforced Neural Reasoning Network module with two output heads.

        Args:
            input_size (int): number of features from input.
            output_size (int): number of outputs.
            layer_sizes (list): A list containing the number of output logics for each layer.
            feature_names (list): A list of feature names.
            n_selected_features_input (int): The number of features to include in each logic in the input layer.
            n_selected_features_internal (int): The number of logics to include in each logic in the internal layers.
            n_selected_features_output (int): The number of logics to include in each logic in the output layer.
            perform_prune_quantile (float): The quantile to use for pruning randomized RN.
            ucb_scale (float): The scale of the confidence interval in the multi-armed bandit policy.
                               c = 1.96 is a 95% confidence interval.
            normal_form (str): 'dnf' for disjunctive normal form network; 'cnf' for conjunctive normal form network.
            add_negations (bool): add negations of logic.
            weight_init (float): Upper bound of uniform weight initialization.  Lower bound is negated value.
        """
        super(TARNRNTraceModule, self).__init__()
        self.ucb_scale = ucb_scale
        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = layer_sizes
        self.head_layer_sizes = head_layer_sizes or []
        self.mlp_head_hidden_dim = int(mlp_head_hidden_dim or 0)
        self.feature_names = feature_names
        self.n_selected_features_input = n_selected_features_input
        self.n_selected_features_internal = n_selected_features_internal
        self.n_selected_features_output = n_selected_features_output
        self.perform_prune_quantile = perform_prune_quantile
        self.normal_form = normal_form
        self.add_negations = add_negations
        self.weight_init = weight_init
        self.logits = logits
        self.logger = logging.getLogger(self.__class__.__name__)

        assert self.normal_form in ['cnf', 'dnf'], "'normal_form' must be one of 'dnf', 'cnf'."

        def _build_logic_layer(layer_index, in_features, out_features, operands, is_input=False):
            if is_input:
                if self.normal_form == 'dnf':
                    return LukasiewiczChannelAndBlock(
                        channels=1,
                        in_features=in_features,
                        out_features=out_features,
                        n_selected_features=n_selected_features_input,
                        parent_weights_dimension='out_features',
                        operands=Predicates(feature_names=feature_names),
                        outputs_key='0',
                        add_negations=add_negations,
                        weight_init=weight_init
                    )
                return LukasiewiczChannelOrBlock(
                    channels=1,
                    in_features=in_features,
                    out_features=out_features,
                    n_selected_features=n_selected_features_input,
                    parent_weights_dimension='out_features',
                    operands=Predicates(feature_names=feature_names),
                    outputs_key='0',
                    add_negations=add_negations,
                    weight_init=weight_init
                )

            if layer_index % 2 == 0:
                if self.normal_form == 'dnf':
                    return LukasiewiczChannelAndBlock(
                        channels=1,
                        in_features=in_features,
                        out_features=out_features,
                        n_selected_features=n_selected_features_internal,
                        parent_weights_dimension='out_features',
                        operands=operands,
                        outputs_key=str(layer_index),
                        weight_init=weight_init
                    )
                return LukasiewiczChannelOrBlock(
                    channels=1,
                    in_features=out_features,
                    out_features=out_features,
                    n_selected_features=n_selected_features_internal,
                    parent_weights_dimension='out_features',
                    operands=operands,
                    outputs_key=str(layer_index),
                    weight_init=weight_init
                )

            if self.normal_form == 'dnf':
                return LukasiewiczChannelOrBlock(
                    channels=1,
                    in_features=out_features,
                    out_features=out_features,
                    n_selected_features=n_selected_features_internal,
                    parent_weights_dimension='out_features',
                    operands=operands,
                    outputs_key=str(layer_index),
                    weight_init=weight_init
                )
            return LukasiewiczChannelAndBlock(
                channels=1,
                in_features=in_features,
                out_features=out_features,
                n_selected_features=n_selected_features_internal,
                parent_weights_dimension='out_features',
                operands=operands,
                outputs_key=str(layer_index),
                weight_init=weight_init
            )

        input_layer = _build_logic_layer(0, input_size, layer_sizes[0], None, is_input=True)

        model_layers = [input_layer]

        # add alternating internal layers
        for i in range(1, len(layer_sizes)):

            internal_layer = _build_logic_layer(i, layer_sizes[i - 1], layer_sizes[i], model_layers[-1])
            model_layers.append(internal_layer)

        self.shared_model = torch.nn.Sequential(*model_layers)

        def _build_head_layers(base_operands, shared_depth):
            head_layers = []
            operands = base_operands
            prev_features = layer_sizes[-1]
            for head_offset, head_size in enumerate(self.head_layer_sizes):
                layer_index = shared_depth + head_offset
                head_layer = _build_logic_layer(layer_index, prev_features, head_size, operands)
                head_layers.append(head_layer)
                operands = head_layer
                prev_features = head_size
            return nn.ModuleList(head_layers), operands, prev_features

        self.head_layers_1, head_operands_1, head_features_1 = _build_head_layers(model_layers[-1], len(layer_sizes))
        self.head_layers_2, head_operands_2, head_features_2 = _build_head_layers(model_layers[-1], len(layer_sizes))

        # Create two output heads
        total_depth = len(layer_sizes) + len(self.head_layer_sizes)
        is_even_layers = total_depth % 2 == 0
        self.output_layer_1 = None
        self.output_layer_2 = None
        self.mlp_head_1 = None
        self.mlp_head_2 = None

        if self.mlp_head_hidden_dim > 0:
            self.mlp_head_1 = nn.Sequential(
                nn.Linear(head_features_1, self.mlp_head_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.mlp_head_hidden_dim, 1),
            )
            self.mlp_head_2 = nn.Sequential(
                nn.Linear(head_features_2, self.mlp_head_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.mlp_head_hidden_dim, 1),
            )
        else:
            # Output head 1
            if is_even_layers:
                if self.normal_form == 'dnf':
                    self.output_layer_1 = LukasiewiczChannelAndBlock(
                        channels=1,
                        in_features=head_features_1,
                        out_features=1,
                        n_selected_features=n_selected_features_output,
                        parent_weights_dimension='out_features',
                        operands=head_operands_1,
                        outputs_key='output_layer_1',
                        weight_init=weight_init
                    )
                else:
                    self.output_layer_1 = LukasiewiczChannelOrBlock(
                        channels=1,
                        in_features=head_features_1,
                        out_features=1,
                        n_selected_features=n_selected_features_output,
                        parent_weights_dimension='out_features',
                        operands=head_operands_1,
                        outputs_key='output_layer_1',
                        weight_init=weight_init
                    )
            else:
                if self.normal_form == 'dnf':
                    self.output_layer_1 = LukasiewiczChannelOrBlock(
                        channels=1,
                        in_features=head_features_1,
                        out_features=1,
                        n_selected_features=n_selected_features_output,
                        parent_weights_dimension='out_features',
                        operands=head_operands_1,
                        outputs_key='output_layer_1',
                        weight_init=weight_init
                    )
                else:
                    self.output_layer_1 = LukasiewiczChannelAndBlock(
                        channels=1,
                        in_features=head_features_1,
                        out_features=1,
                        n_selected_features=n_selected_features_output,
                        parent_weights_dimension='out_features',
                        operands=head_operands_1,
                        outputs_key='output_layer_1',
                        weight_init=weight_init
                    )

            # Output head 2 (identical structure to head 1)
            if is_even_layers:
                if self.normal_form == 'dnf':
                    self.output_layer_2 = LukasiewiczChannelAndBlock(
                        channels=1,
                        in_features=head_features_2,
                        out_features=1,
                        n_selected_features=n_selected_features_output,
                        parent_weights_dimension='out_features',
                        operands=head_operands_2,
                        outputs_key='output_layer_2',
                        weight_init=weight_init
                    )
                else:
                    self.output_layer_2 = LukasiewiczChannelOrBlock(
                        channels=1,
                        in_features=head_features_2,
                        out_features=1,
                        n_selected_features=n_selected_features_output,
                        parent_weights_dimension='out_features',
                        operands=head_operands_2,
                        outputs_key='output_layer_2',
                        weight_init=weight_init
                    )
            else:
                if self.normal_form == 'dnf':
                    self.output_layer_2 = LukasiewiczChannelOrBlock(
                        channels=1,
                        in_features=head_features_2,
                        out_features=1,
                        n_selected_features=n_selected_features_output,
                        parent_weights_dimension='out_features',
                        operands=head_operands_2,
                        outputs_key='output_layer_2',
                        weight_init=weight_init
                    )
                else:
                    self.output_layer_2 = LukasiewiczChannelAndBlock(
                        channels=1,
                        in_features=head_features_2,
                        out_features=1,
                        n_selected_features=n_selected_features_output,
                        parent_weights_dimension='out_features',
                        operands=head_operands_2,
                        outputs_key='output_layer_2',
                        weight_init=weight_init
                    )

        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, x: torch.Tensor):
        shared_x = self.shared_model(x)
        head_1_x = shared_x
        for layer in self.head_layers_1:
            head_1_x = layer(head_1_x)
        head_2_x = shared_x
        for layer in self.head_layers_2:
            head_2_x = layer(head_2_x)

        if self.mlp_head_hidden_dim > 0:
            head_1_inputs = head_1_x.squeeze(1) if head_1_x.dim() == 3 else head_1_x
            head_2_inputs = head_2_x.squeeze(1) if head_2_x.dim() == 3 else head_2_x
            output_1 = self.mlp_head_1(head_1_inputs).squeeze(-1)
            output_2 = self.mlp_head_2(head_2_inputs).squeeze(-1)
        else:
            if self.logits:
                head_1_outputs = torch.special.logit(self.output_layer_1(head_1_x), eps=1e-6)
                head_2_outputs = torch.special.logit(self.output_layer_2(head_2_x), eps=1e-6)
            else:
                head_1_outputs = self.output_layer_1(head_1_x)
                head_2_outputs = self.output_layer_2(head_2_x)

            output_1 = head_1_outputs.mean(dim=(1, 2))
            output_2 = head_2_outputs.mean(dim=(1, 2))

        return output_1, output_2

    ROUNDING_PRECISION = 32
    EPS = 1e-2

    def _get_head_output_layer(self, output_channel: int):
        if self.mlp_head_hidden_dim > 0:
            raise NotImplementedError("Logical explanations are unavailable when mlp_head_hidden_dim > 0.")
        if output_channel == 0:
            return self.output_layer_1
        if output_channel == 1:
            return self.output_layer_2
        raise ValueError("output_channel must be 0 or 1")

    def explain_samples(
            self,
            x: torch.Tensor,
            quantile: float = 0.5,
            threshold: float = None,
            target_names: list = None,
            explain_type: str = 'both',
            print_type: str = 'logical',
            sample_explanation_prefix: str = "The sample has",
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform_features=None,
            inverse_transform_target=None,
            show_bounds: bool = True,
            simplify: bool = False,
            exclusions: list[str] = None,
            min_max_feature_dict: dict = None,
            feature_importances: bool = False,
            feature_importances_type: str = 'weight',
            output_channel: int = 0,
            logic_channel: Optional[int] = None
    ) -> str:
        x = x.to(next(self.parameters()).device)
        output_layer = self._get_head_output_layer(output_channel)
        if logic_channel is None:
            logic_channel = 0
        if target_names is None:
            target_names = ['control_outcome', 'treated_outcome']
        sample_explanations = []
        for i in range(x.size(0)):
            outputs = {}
            register_hooks(self, outputs)
            predictions = self.forward(x[i].unsqueeze(0))
            required_output_threshold = predictions[output_channel][0].detach().cpu().squeeze()
            prediction_value = required_output_threshold
            force_negate = prediction_value < 0.5

            explanation_str = ''
            quantile_candidates = [quantile, 0.95, 0.9, 0.8, 0.7, 0.5]
            ignore_candidates = [ignore_uninformative, False]
            for q in quantile_candidates:
                for ignore_flag in ignore_candidates:
                    op_explain = output_layer.explain_sample(
                        outputs_dict=outputs,
                        required_output_thresholds=min(
                            required_output_threshold * (1 + self.EPS if force_negate else 1 - self.EPS), 1.0),
                        quantile=q,
                        threshold=threshold,
                        parent_weights=torch.tensor(1.),
                        parent_logic_type='And',
                        depth=0,
                        explain_type=explain_type,
                        print_type=print_type if print_type == 'natural' else 'logical',
                        input_features=x[i].cpu(),
                        channel=logic_channel,
                        force_negate=force_negate,
                        ignore_uninformative=ignore_flag,
                        rounding_precision=self.ROUNDING_PRECISION,
                        inverse_transform=inverse_transform_features,
                        show_bounds=show_bounds,
                        original_rounding_precision=rounding_precision,
                    )
                    explanation_str = ', '.join(np.unique(op_explain).tolist())
                    if explanation_str:
                        break
                if explanation_str:
                    break
            if not explanation_str:
                raise RuntimeError('Could not produce explanation for selected head.')

            if inverse_transform_target is not None:
                prediction_value = inverse_transform_target(np.array(prediction_value).reshape(1, -1))

            if print_type in ['logical', 'logical-natural'] or feature_importances:
                explanation_tree = simplification(
                    explanation_str,
                    print_type,
                    simplify=simplify,
                    sample_level=True,
                    ndigits=rounding_precision,
                    exclusions=exclusions,
                    min_max_feature_dict=min_max_feature_dict,
                    feature_importances=feature_importances,
                    verbose=False
                )
                explanation_str = explanation_tree.tree_to_string()

            target_label = target_names[output_channel] if len(target_names) > output_channel else f'output_{output_channel}'
            sample_explanations.append(
                f"{i}: {sample_explanation_prefix} a predicted {target_label} of {round(float(prediction_value), rounding_precision)} because: \n\n{explanation_str}"
            )

        sample_explanations = '\n'.join(sample_explanations)
        if print_type == 'natural':
            sample_explanations = '.  '.join([x[:1].capitalize() + x[1:] for x in sample_explanations.split('.  ')])
        return sample_explanations

    def explain(
            self,
            quantile: float = 0.5,
            required_output_thresholds: torch.Tensor = torch.tensor(0.9),
            threshold: float = None,
            explain_type: str = 'both',
            print_type: str = 'logical',
            target_names: list = None,
            explanation_prefix: str = "A sample has a",
            ignore_uninformative: bool = False,
            rounding_precision: int = 3,
            inverse_transform_features=None,
            inverse_transform_target=None,
            show_bounds: bool = True,
            simplify: bool = False,
            exclusions: list[str] = None,
            output_channel: int = 0,
            logic_channel: Optional[int] = None
    ) -> str:
        output_layer = self._get_head_output_layer(output_channel)
        if logic_channel is None:
            logic_channel = 0
        force_negate = required_output_thresholds < 0.5
        if target_names is None:
            target_names = ['control_outcome', 'treated_outcome']
        op_explain = output_layer.explain(
            quantile=quantile,
            required_output_thresholds=required_output_thresholds,
            threshold=threshold,
            parent_weights=torch.tensor(1.),
            parent_logic_type='Or',
            depth=0,
            explain_type=explain_type,
            print_type=print_type if print_type == 'natural' else 'logical',
            channel=logic_channel,
            force_negate=force_negate,
            ignore_uninformative=ignore_uninformative,
            rounding_precision=self.ROUNDING_PRECISION,
            inverse_transform=inverse_transform_features,
            show_bounds=show_bounds,
            original_rounding_precision=rounding_precision,
        )
        explanation_str = ', '.join(np.unique(op_explain).tolist())
        if not explanation_str:
            raise RuntimeError('Could not produce global explanation for selected head.')
        display_threshold = required_output_thresholds
        if inverse_transform_target is not None:
            display_threshold = inverse_transform_target(np.array(required_output_thresholds).reshape(1, -1))
        explanation_str = simplification(explanation_str, print_type, simplify, sample_level=False, ndigits=rounding_precision, exclusions=exclusions)
        target_label = target_names[output_channel] if len(target_names) > output_channel else f'output_{output_channel}'
        return f"{explanation_prefix} predicted {target_label} of {round(float(display_threshold), rounding_precision)} because: \n\n{explanation_str}"




