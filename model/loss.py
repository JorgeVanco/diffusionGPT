import torch
from torch import Tensor
import torch.nn.functional as F


def cdcd_loss(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """
    Compute cross-entropy loss only on masked (noisy) positions.
    
    Args:
        logits: [B, L, V] predicted logits
        target: [B, L] target token indices
        mask: [B, L] binary mask (1 = compute loss, 0 = ignore)
    
    Returns:
        Scalar loss value
    """
    B, L, V = logits.shape
    
    # Flatten for cross entropy
    logits_flat = logits.view(-1, V)  # [B*L, V]
    target_flat = target.view(-1)  # [B*L]
    
    # Compute cross entropy without reduction
    loss_per_token = F.cross_entropy(logits_flat, target_flat, reduction='none')  # [B*L]
    
    # Reshape back and apply mask
    loss_per_token = loss_per_token.view(B, L)  # [B, L]
    mask_flat = mask.view(B, L) if mask.dim() == 3 else mask
    
    # Only compute loss on masked positions
    masked_loss = loss_per_token * mask_flat
    
    # Average over masked positions only
    num_masked = mask_flat.sum()
    if num_masked > 0:
        loss = masked_loss.sum() / num_masked
    else:
        loss = masked_loss.sum()  # Fallback (shouldn't happen)
    
    return loss