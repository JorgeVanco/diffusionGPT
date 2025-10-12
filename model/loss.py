import torch
from torch import Tensor
import torch.nn.functional as F

def cdcd_loss(logits: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    return F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))