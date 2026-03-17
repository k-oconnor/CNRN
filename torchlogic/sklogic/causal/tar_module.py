import torch
import torch.nn as nn
from torchlogic.modules.tarn import TARNRNModule

class TARNet(nn.Module):
    def __init__(
        self,
        input_size,
        layer_sizes,
        head_layer_sizes,
        feature_names,
        n_selected_features_input,
        n_selected_features_internal,
        n_selected_features_output,
        perform_prune_quantile,
        ucb_scale,
        normal_form='cnf',
        add_negations=False,
        weight_init=0.2

    ):
        super(TARNet, self).__init__()
        self.symbolic_model = TARNRNModule(
            input_size=input_size,
            output_size=2,  # Two heads: y0, y1
            layer_sizes=layer_sizes,
            head_layer_sizes=head_layer_sizes,
            feature_names=feature_names,
            n_selected_features_input=n_selected_features_input,
            n_selected_features_internal=n_selected_features_internal,
            n_selected_features_output=n_selected_features_output,
            perform_prune_quantile=perform_prune_quantile,
            ucb_scale=ucb_scale,
            normal_form=normal_form,
            add_negations=add_negations,
            weight_init=weight_init

        )

    def forward(self, x):
        q_t0, q_t1 = self.symbolic_model(x)
        return q_t0, q_t1

    def explain_samples(self, *args, **kwargs):
        return self.symbolic_model.explain_samples(*args, **kwargs)

    def explain(self, *args, **kwargs):
        return self.symbolic_model.explain(*args, **kwargs)

