from copy import deepcopy

import torch
from torch import nn

from .brn import BanditNRNModule


class LogicJEPA(nn.Module):
    """
    JEPA-style encoder built from NRN blocks.

    The online encoder is trained directly. The target encoder is an EMA copy
    updated by `update_target_encoder`.
    """

    def __init__(
            self,
            input_size: int,
            feature_names: list,
            layer_sizes: list,
            embedding_dim: int,
            predictor_hidden_dim: int,
            predictor_depth: int,
            n_selected_features_input: int,
            n_selected_features_internal: int,
            n_selected_features_output: int,
            perform_prune_quantile: float,
            ucb_scale: float,
            normal_form: str = 'cnf',
            add_negations: bool = False,
            weight_init: float = 0.2
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)

        self.online_encoder = BanditNRNModule(
            input_size=input_size,
            output_size=self.embedding_dim,
            layer_sizes=layer_sizes,
            feature_names=feature_names,
            n_selected_features_input=n_selected_features_input,
            n_selected_features_internal=n_selected_features_internal,
            n_selected_features_output=n_selected_features_output,
            perform_prune_quantile=perform_prune_quantile,
            ucb_scale=ucb_scale,
            normal_form=normal_form,
            add_negations=add_negations,
            weight_init=weight_init,
            logits=False
        )
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)

        depth = max(1, int(predictor_depth))
        layers = []
        in_dim = self.embedding_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, predictor_hidden_dim))
            layers.append(nn.GELU())
            in_dim = predictor_hidden_dim
        layers.append(nn.Linear(in_dim, self.embedding_dim))
        self.predictor = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor, use_target: bool = False):
        encoder = self.target_encoder if use_target else self.online_encoder
        z = encoder(x)
        return nn.functional.normalize(z, dim=-1)

    def forward_views(self, context_view: torch.Tensor, target_view: torch.Tensor):
        z_online = self.encode(context_view, use_target=False)
        z_target = self.encode(target_view, use_target=True)
        z_pred = nn.functional.normalize(self.predictor(z_online), dim=-1)
        return z_online, z_pred, z_target

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.99):
        momentum = float(momentum)
        for target_param, online_param in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)
