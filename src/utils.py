import torch

def mask_input_ids_(input_ids: torch.Tensor, mask_token_id: int, mask_prob: torch.Tensor, remasking_mask: torch.Tensor | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """In-place mask input_ids with given mask probability.
    At each position, with probability mask_prob, replace with mask_token_id.
    Args:
        input_ids (torch.Tensor): Tensor of shape (B, L) containing token ids.
        mask_token_id (int): The token id used for masking.
        mask_prob (torch.Tensor): Tensor of shape (B,) with masking probabilities for each example.
        remasking_mask (torch.Tensor, optional): Boolean tensor of shape (B, L) indicating positions eligible for re-masking.
        generator (torch.Generator, optional): Random generator for reproducibility.
    Returns:
        torch.Tensor: A boolean tensor of shape (B, L) indicating which positions were masked.
    """
    prob_matrix = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    mask_matrix = (prob_matrix < mask_prob.view(-1, 1))
    if remasking_mask is not None:
        mask_matrix = mask_matrix & remasking_mask

    input_ids[mask_matrix] = mask_token_id
    return mask_matrix
    
    
def _dispatch_table_logging(self, content, step, trainer) -> None:
    # --- WANDB ---
    # Check if WandB is enabled in 'report_to'
    if "wandb" in trainer.args.report_to:
        import wandb
        # WandB expects a Table for text or Image for images
        table = wandb.Table(columns=["Prompt", "Generated"])
        for p, g in zip(self.prompts, content):
            table.add_data(p, g)
        wandb.log({"evaluation_samples": table}, step=step, commit=False)

    # --- TENSORBOARD ---
    if "tensorboard" in trainer.args.report_to:
        from transformers.integrations.integration_utils import TensorBoardCallback
        # Finding the TB writer is a bit tricky, it's hidden inside the trainer
        tb_callback = [c for c in trainer.callback_handler.callbacks if isinstance(c, TensorBoardCallback)]
        if tb_callback and tb_callback[0].tb_writer is not None:
            writer = tb_callback[0].tb_writer
            # TensorBoard expects specific 'add_text' calls
            for i, (p, g) in enumerate(zip(self.prompts, content)):
                writer.add_text(f"gen_sample_{i}", f"**Prompt:** {p}  \n**Gen:** {g}", step)

    # --- MLFLOW ---
    if "mlflow" in trainer.args.report_to:
        import mlflow # type: ignore
        # MLFlow usually logs artifacts (files) or text
        for i, (p, g) in enumerate(zip(self.prompts, content)):
            mlflow.log_text(g, f"step_{step}_sample_{i}.txt")