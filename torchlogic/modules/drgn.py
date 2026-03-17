import torch
from torch import nn
import logging

from ..nn import LukasiewiczChannelOrBlock, LukasiewiczChannelAndBlock, Predicates


class DragonNRNModule(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        layer_sizes,
        feature_names,
        n_selected_features_input,
        n_selected_features_internal,
        n_selected_features_output,
        perform_prune_quantile,
        ucb_scale,
        normal_form='dnf',
        add_negations=False,
        weight_init=0.2,
        propensity_branch_layer=1  # NEW: Where propensity branches off
    ):
        """
        Proper DragonNet architecture with early branching for propensity.
        
        Args:
            propensity_branch_layer: Layer index where propensity branches.
                                    0 = branch from input (most independent)
                                    1 = branch after first layer (recommended)
                                    len(layer_sizes) = branch from final (least independent)
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.normal_form = normal_form
        self.output_size = output_size
        self.layer_sizes = layer_sizes
        self.propensity_branch_layer = min(propensity_branch_layer, len(layer_sizes) - 1)

        assert normal_form in ['dnf', 'cnf'], "'normal_form' must be 'dnf' or 'cnf'."
        assert len(layer_sizes) > 0, "layer_sizes must not be empty"

        # Input Layer - Start with AND for DNF, OR for CNF
        if normal_form == 'dnf':
            input_layer = LukasiewiczChannelAndBlock(
                channels=output_size,
                in_features=input_size,
                out_features=layer_sizes[0],
                n_selected_features=n_selected_features_input,
                parent_weights_dimension='out_features',
                operands=Predicates(feature_names=feature_names),
                outputs_key='layer_0',
                add_negations=add_negations,
                weight_init=weight_init
            )
        else:  # CNF
            input_layer = LukasiewiczChannelOrBlock(
                channels=output_size,
                in_features=input_size,
                out_features=layer_sizes[0],
                n_selected_features=n_selected_features_input,
                parent_weights_dimension='out_features',
                operands=Predicates(feature_names=feature_names),
                outputs_key='layer_0',
                add_negations=add_negations,
                weight_init=weight_init
            )

        model_layers = [input_layer]

        # Internal Layers - Proper alternation
        for i in range(1, len(layer_sizes)):
            prev_layer = model_layers[-1]
            outputs_key = f'layer_{i}'

            # Fixed alternation logic:
            # For DNF: AND -> OR -> AND -> OR ...
            # For CNF: OR -> AND -> OR -> AND ...
            if normal_form == 'dnf':
                use_and_block = (i % 2 == 0)
            else:  # CNF
                use_and_block = (i % 2 == 1)

            if use_and_block:
                block_cls = LukasiewiczChannelAndBlock
            else:
                block_cls = LukasiewiczChannelOrBlock

            internal_layer = block_cls(
                channels=output_size,
                in_features=layer_sizes[i - 1],
                out_features=layer_sizes[i],
                n_selected_features=n_selected_features_internal,
                parent_weights_dimension='out_features',
                operands=prev_layer,
                outputs_key=outputs_key,
                weight_init=weight_init
            )
            model_layers.append(internal_layer)

        # Shared Trunk (all layers)
        self.shared_trunk = nn.Sequential(*model_layers)
        
        # Store branching point for propensity
        self.branch_layer = model_layers[self.propensity_branch_layer]
        branch_features = layer_sizes[self.propensity_branch_layer]
        
        # Store final layer for outcome heads
        final_layer = model_layers[-1]
        final_features = layer_sizes[-1]

        # === PROPENSITY HEAD (branches early) ===
        # Uses simpler pathway from earlier representations
        # Single channel for binary classification task
        
        # Propensity-specific intermediate layer (if needed)
        if self.propensity_branch_layer < len(layer_sizes) - 1:
            # Add one intermediate layer for propensity
            self.prop_intermediate = LukasiewiczChannelOrBlock(
                channels=1,  # Single channel for binary task
                in_features=branch_features,
                out_features=max(4, branch_features // 2),
                n_selected_features=min(n_selected_features_internal, branch_features),
                parent_weights_dimension='out_features',
                operands=self.branch_layer,
                outputs_key='prop_intermediate',
                weight_init=weight_init * 1.5
            )
            prop_in_features = max(4, branch_features // 2)
            prop_operands = self.prop_intermediate
        else:
            # Branch from final layer
            self.prop_intermediate = None
            prop_in_features = branch_features
            prop_operands = self.branch_layer
        
        # Propensity output head
        self.head_g = LukasiewiczChannelAndBlock(
            channels=1,
            in_features=prop_in_features,
            out_features=1,
            n_selected_features=min(n_selected_features_output, prop_in_features),
            parent_weights_dimension='out_features',
            operands=prop_operands,
            outputs_key='head_g',
            weight_init=weight_init * 1.5
        )

        # === OUTCOME HEADS (use full deep representations) ===
        # These get the benefit of the full trunk
        self.head_f0 = LukasiewiczChannelAndBlock(
            channels=output_size,
            in_features=final_features,
            out_features=1,
            n_selected_features=n_selected_features_output,
            parent_weights_dimension='out_features',
            operands=final_layer,
            outputs_key='head_f0',
            weight_init=weight_init
        )

        self.head_f1 = LukasiewiczChannelAndBlock(
            channels=output_size,
            in_features=final_features,
            out_features=1,
            n_selected_features=n_selected_features_output,
            parent_weights_dimension='out_features',
            operands=final_layer,
            outputs_key='head_f1',
            weight_init=weight_init
        )

        # Need to figure out how to handle pruning and UCB scaling
        self.perform_prune_quantile = perform_prune_quantile
        self.ucb_scale = ucb_scale
        
        self.logger.info(f"DragonNet architecture: Propensity branches at layer {self.propensity_branch_layer}")

    def forward(self, x: torch.Tensor):
        # Forward pass through shared trunk
        trunk_output = self.shared_trunk(x)
        
        # Get branching point output for propensity
        # (This is cached during trunk forward pass)
        if self.propensity_branch_layer == len(self.layer_sizes) - 1:
            branch_output = trunk_output
        else:
            # Re-forward through trunk up to branch point
            branch_output = x
            for i in range(self.propensity_branch_layer + 1):
                branch_output = self.shared_trunk[i](branch_output)
        
        # Propensity pathway (branches early)
        if self.prop_intermediate is not None:
            prop_features = self.prop_intermediate(branch_output)
        else:
            prop_features = branch_output
        g = self.head_g(prop_features).squeeze(-1).squeeze(-1)
        
        # Outcome heads (use full trunk)
        q_t0 = self.head_f0(trunk_output).mean(dim=(1, 2))
        q_t1 = self.head_f1(trunk_output).mean(dim=(1, 2))
        
        return q_t0, q_t1, g

    def get_layer_info(self):
        """Debug method to inspect layer structure"""
        info = []
        info.append(f"Normal form: {self.normal_form}")
        info.append(f"Layer sizes: {self.layer_sizes}")
        info.append(f"Propensity branches at layer: {self.propensity_branch_layer}")
        
        info.append("\nShared Trunk:")
        for i, layer in enumerate(self.shared_trunk):
            layer_type = "AND" if isinstance(layer, LukasiewiczChannelAndBlock) else "OR"
            branch_marker = " <- PROPENSITY BRANCHES HERE" if i == self.propensity_branch_layer else ""
            info.append(f"  Layer {i}: {layer_type} - {layer.in_features} -> {layer.out_features}{branch_marker}")
        
        info.append(f"\nPropensity pathway (from layer {self.propensity_branch_layer}):")
        if self.prop_intermediate:
            info.append(f"  Intermediate: OR - {self.prop_intermediate.in_features} -> {self.prop_intermediate.out_features}")
        info.append(f"  Output: AND - {self.head_g.in_features} -> {self.head_g.out_features}")
        
        info.append(f"\nOutcome pathways (from final layer):")
        info.append(f"  Y0 head: AND - {self.head_f0.in_features} -> {self.head_f0.out_features}")
        info.append(f"  Y1 head: AND - {self.head_f1.in_features} -> {self.head_f1.out_features}")
        
        return "\n".join(info)

    def init_input_mask_from_policy(self, policy_probs: torch.Tensor, retain_ratio: float = 1.0):
        """Initialize the input layer's feature selection mask using provided probabilities.

        Args:
            policy_probs: 1D tensor of shape [input_size] with non-negative scores/probabilities per input feature.
            retain_ratio: fraction of selections to take strictly from top-probability features (0<r<=1).
        """
        try:
            input_layer = self.shared_trunk[0]
        except Exception:
            return

        # Validate shape
        if policy_probs is None:
            return
        policy = policy_probs.detach().float().clone()
        if policy.dim() != 1 or policy.numel() != input_layer.in_features:
            self.logger.warning("Policy size mismatch; skipping MIC-based mask init")
            return

        # Normalize to probabilities (avoid zeros)
        eps = 1e-8
        policy = policy.clamp(min=0)
        policy = policy / (policy.sum() + eps)

        channels = input_layer.channels
        out_features = input_layer.out_features
        k = input_layer.n_selected_features

        # Determine number of retained top features vs sampled
        top_k = max(1, int(round(k * float(retain_ratio))))
        rem_k = max(0, k - top_k)

        # Precompute top indices
        top_indices = torch.topk(policy, k=k, largest=True).indices
        top_fixed = top_indices[:top_k]

        # Prepare mask tensor [channels, out_features, n_selected_features]
        mask = torch.empty((channels, out_features, k), dtype=torch.long)

        # Sampling distribution for remaining slots (exclude already chosen)
        base_probs = policy.clone()

        for c in range(channels):
            for o in range(out_features):
                if rem_k > 0:
                    probs = base_probs.clone()
                    probs[top_fixed] = 0.0
                    probs = probs / (probs.sum() + eps)
                    sampled = torch.multinomial(probs, num_samples=rem_k, replacement=False)
                    sel = torch.cat([top_fixed, sampled], dim=0)
                else:
                    sel = top_fixed.clone()
                mask[c, o] = sel

        # Handle negations doubling
        if getattr(input_layer, 'add_negations', False):
            mask = torch.cat([mask, mask.clone()], dim=-1)

        # Copy into module buffer
        input_layer.mask.copy_(mask.to(input_layer.mask.device))

    def refresh_input_mask(self, policy_probs: torch.Tensor, retain_ratio: float = 0.8, swap_frac: float = 0.2):
        """Refresh input layer mask by retaining a portion of current selections and swapping the rest
        according to policy probabilities.

        Args:
            policy_probs: 1D tensor of shape [input_size] with non-negative scores/probabilities per input feature.
            retain_ratio: fraction of current selected features to retain per unit (0<r<=1).
            swap_frac: fraction of total selected features to replace with samples from policy (0<=f<=1).
        """
        try:
            input_layer = self.shared_trunk[0]
        except Exception:
            return

        if policy_probs is None:
            return
        policy = policy_probs.detach().float().clone()
        if policy.dim() != 1 or policy.numel() != input_layer.in_features:
            self.logger.warning("Policy size mismatch; skipping mask refresh")
            return

        eps = 1e-8
        policy = policy.clamp(min=0)
        policy = policy / (policy.sum() + eps)

        channels = input_layer.channels
        out_features = input_layer.out_features
        k = input_layer.n_selected_features

        retain_k = max(0, min(k, int(round(k * float(retain_ratio) * (1.0 - float(swap_frac))))))
        replace_k = k - retain_k

        cur_mask = input_layer.mask.detach().clone()  # [channels, out_features, k] (or doubled with negations)
        # If negations are enabled, the mask is doubled; use only first k slots as canonical indices
        if getattr(input_layer, 'add_negations', False) and cur_mask.size(-1) >= 2 * k:
            cur_mask_k = cur_mask[..., :k]
        else:
            cur_mask_k = cur_mask

        new_mask = torch.empty_like(cur_mask_k)

        for c in range(channels):
            for o in range(out_features):
                current_sel = cur_mask_k[c, o]
                if retain_k > 0:
                    retained = current_sel[:retain_k]
                else:
                    retained = torch.empty(0, dtype=torch.long)
                if replace_k > 0:
                    probs = policy.clone()
                    # avoid duplicates
                    probs[retained] = 0.0
                    probs = probs / (probs.sum() + eps)
                    sampled = torch.multinomial(probs, num_samples=replace_k, replacement=False)
                    sel = torch.cat([retained, sampled], dim=0)
                else:
                    sel = retained
                new_mask[c, o] = sel

        # Re-apply negations duplication if present
        if getattr(input_layer, 'add_negations', False):
            new_mask = torch.cat([new_mask, new_mask.clone()], dim=-1)

        input_layer.mask.copy_(new_mask.to(input_layer.mask.device))
