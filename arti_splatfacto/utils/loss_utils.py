import torch

def opacity_loss(opacity, visibility_filter=None):
    """
    Encourage opacity values to be close to 0 or 1 (sparse, non-transparent Gaussians).
    
    Args:
        opacity (Tensor): Predicted opacity values, shape (N,) or (N,1)
        visibility_filter (Tensor, optional): Boolean mask of visible Gaussians (True = visible)
    """
    opacity = opacity.clamp(1e-6, 1-1e-6) # Prevent log(0)
    log_opacity = opacity * torch.log(opacity)
    log_one_minus_opacity = (1 - opacity) * torch.log(1 - opacity)
    sparse_loss = -1 * (log_opacity + log_one_minus_opacity)
    
    if visibility_filter is not None:
        sparse_loss = sparse_loss[visibility_filter]
    
    return sparse_loss.mean()
