from dataclasses import dataclass
from turtle import distance
from typing import Any, Dict, Tuple, Union
import torch
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate, remove, reset_opa, split


@dataclass 
class SpatialArtiStrategy(DefaultStrategy):
    """
    new densification strategy with spatial priors for ARTI-SPLATFACTO.
    """
    def __init__(self, owner=None, distance_prune_start=5000, distance_threshold=0.05, 
                 prune_probability=0.6, grad_zero_threshold=0.1, **kwargs):
        super().__init__(**kwargs)
        self.owner = owner  # Object3DSeg instance
        self.distance_prune_start = distance_prune_start
        self.distance_threshold = distance_threshold 
        self.prune_probability = prune_probability   
        self.grad_zero_threshold = grad_zero_threshold  
        self.grow_probability = 0.5

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """post-backward with staged pruning strategy."""
        if step >= self.refine_stop_iter:
            return
        
        if step >= self.distance_prune_start and self.owner is not None:
            self.gradient_mask(params, state, step, info)
        
        self._update_state(params, state, info, packed=packed)

        if step >= self.distance_prune_start:
            self.distance_opacity_decay(params, step)
        
        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            if step < self.distance_prune_start:
                # Early training: standard growth + standard pruning
                n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
                if self.verbose:
                    print(f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(params['means'])} GSs.")
                
                n_prune = self._prune_gs(params, optimizers, state, step)
                if self.verbose:
                    print(f"Step {step}: {n_prune} GSs pruned (standard). "
                        f"Now having {len(params['means'])} GSs.")
            else:
                # Later training: probabilistic growth +  pruning
                n_dupli, n_split = self._grow_gs_probabilistic(params, optimizers, state, step)
                if self.verbose:
                    print(f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split (probabilistic). "
                        f"Now having {len(params['means'])} GSs.")
                
                n_prune = self._prune_gs_hybrid(params, optimizers, state, step)
                if self.verbose:
                    print(f"Step {step}: {n_prune} GSs pruned (hybrid). "
                        f"Now having {len(params['means'])} GSs.")
            
            # Reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()
        
        if step % self.reset_every == 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    def gradient_mask(self, params, state, step, info):
        if state["grad2d"] is None:
            return
            
        means = params["means"]
        distances = self.owner.query_mask_distance(means)
        
        # Zero gradients for Gaussians that are too far
        far_mask = distances > self.grad_zero_threshold
        n_masked = far_mask.sum().item()
        
        if n_masked > 0:
            state["grad2d"][far_mask] = 0.0
            if self.verbose and step % 500 == 0:  
                print(f"Zeroed gradients for {n_masked} far Gaussians (>{self.grad_zero_threshold:.3f})")
    
    def _prune_gs_hybrid(self, params, optimizers, state, step):
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d
            is_prune = is_prune | is_too_big
        
        scales = torch.exp(params["scales"])  
        max_scale_per_gaussian = scales.max(dim=-1)[0]
        max_allowed_scale = 0.1 
        is_too_large = max_scale_per_gaussian > max_allowed_scale
        is_prune = is_prune | is_too_large
        
        n_scale_pruned = is_too_large.sum().item()
        if self.verbose and n_scale_pruned > 0:
            print(f"Scale-pruned {n_scale_pruned} oversized Gaussians (>{max_allowed_scale:.2f})")
        
        if self.owner is not None:
            means = params["means"]
            distances = self.owner.query_mask_distance(means)  
            
            distant_mask = distances > self.distance_threshold
            n_distant = distant_mask.sum().item()
            
            if n_distant > 0:
                outside_distances = distances[distant_mask]
  
                max_prob = self.prune_probability  
                scale = 0.3                        
                distance_probs = max_prob * torch.sigmoid((outside_distances - self.distance_threshold) / scale)
                
                rand_vals = torch.rand_like(distance_probs)
                probabilistic_prune = rand_vals < distance_probs
                
                # Add to pruning mask
                distant_indices = torch.where(distant_mask)[0]
                selected_indices = distant_indices[probabilistic_prune]
                distance_prune_mask = torch.zeros_like(is_prune, dtype=torch.bool)
                distance_prune_mask[selected_indices] = True
                
                is_prune = is_prune | distance_prune_mask
                
                if self.verbose:
                    print(f"Distance-pruned {probabilistic_prune.sum().item()}/{n_distant} distant GSs "
                        f"(max_prob={max_prob}, scale={scale})")

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune

    def _grow_gs_probabilistic(self, params, optimizers, state, step):
        """Probabilistic growth to prevent excessive densification after cleanup phase."""
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        
        if is_dupli.sum() > 0:
            dupli_probs = torch.rand(is_dupli.sum(), device=device)
            probabilistic_dupli = dupli_probs < self.grow_probability  
            
            dupli_indices = torch.where(is_dupli)[0]
            selected_dupli_indices = dupli_indices[probabilistic_dupli]
            is_dupli_filtered = torch.zeros_like(is_dupli, dtype=torch.bool)
            is_dupli_filtered[selected_dupli_indices] = True
            
            n_dupli = is_dupli_filtered.sum().item()
            if n_dupli > 0:
                duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli_filtered)
        else:
            n_dupli = 0

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        
        if is_split.sum() > 0:
            split_probs = torch.rand(is_split.sum(), device=device)
            probabilistic_split = split_probs < self.grow_probability
            
            split_indices = torch.where(is_split)[0]
            selected_split_indices = split_indices[probabilistic_split]
            is_split_filtered = torch.zeros_like(is_split, dtype=torch.bool)
            is_split_filtered[selected_split_indices] = True
            

            is_split_filtered = torch.cat([
                is_split_filtered,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ])
            
            n_split = is_split_filtered.sum().item()
            if n_split > 0:
                split(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    mask=is_split_filtered,
                    revised_opacity=self.revised_opacity,
                )
        else:
            n_split = 0
        
        return n_dupli, n_split
    

    def distance_opacity_decay(self, params, step):
        """Decay opacity of far Gaussians smoothly based on distance."""
        if self.owner is None:
            return

        means = params["means"]
        distances = self.owner.query_mask_distance(means) 

        outside_mask = distances > self.distance_threshold
        if outside_mask.sum() == 0:
            return

        max_reduction = 0.9   
        scale = 0.3           
        normalized_dist = (distances[outside_mask] - self.distance_threshold) / scale
        decay_factor = 1.0 - max_reduction * torch.sigmoid(normalized_dist)

        with torch.no_grad():
            current_opacities = torch.sigmoid(params["opacities"])
            current_opacities[outside_mask] *= decay_factor.unsqueeze(-1)
            params["opacities"].data[outside_mask] = torch.logit(
                torch.clamp(current_opacities[outside_mask], 1e-6, 1-1e-6)
            )

