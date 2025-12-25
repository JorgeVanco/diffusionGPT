import torch

def mask_input_ids_(input_ids, mask_token_id, mask_prob, generator=None) -> None:
    """In-place mask input_ids with given mask probability.
    At each position, with probability mask_prob, replace with mask_token_id.
    Args:
        input_ids (torch.Tensor): Tensor of shape (B, L) containing token ids.
        mask_token_id (int): The token id used for masking.
        mask_prob (torch.Tensor): Tensor of shape (B,) with masking probabilities for each example.
        generator (torch.Generator, optional): Random generator for reproducibility.
    """
    prob_matrix = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    mask_matrix = prob_matrix < mask_prob.view(-1, 1)
    input_ids[mask_matrix] = mask_token_id