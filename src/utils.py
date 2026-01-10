import torch
from transformers import (
    HfArgumentParser, 
)
from typing import Optional, Dict, Any
import sys

from src.argument_classes import DiffusionTrainingArguments, ModelArguments, DataArguments


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
        mask_matrix = mask_matrix & remasking_mask.bool()

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
            

def visualize_diffusion_steps(pipeline_output, tokenizer, sample_idx=0):
    """
    Visualizes the discrete diffusion process in the terminal with colors.
    
    Args:
        pipeline_output: The dict returned by pipeline._forward()
        tokenizer: The tokenizer used for decoding
        sample_idx: Which example in the batch to visualize (default 0)
    """
    # ANSI Color Codes
    GREEN = '\033[92m'
    RED = '\033[91m' # For Masks
    RESET = '\033[0m'
    GRAY = '\033[90m'

    history = pipeline_output["history"]
    
    print(f"\n{GRAY}{'='*20} DIFFUSION PROCESS {'='*20}{RESET}")
    
    # Get the mask ID for comparison
    mask_id = tokenizer.mask_token_id
    
    for step, state_tensor in enumerate(history):
        # state_tensor is (Batch, Seq_Len) -> Get specific sample
        current_ids = state_tensor[sample_idx]
        
        # Decode logic with highlighting
        decoded_tokens = []
        for i, token_id in enumerate(current_ids):
            token_str = tokenizer.decode([token_id])
            
            # 1. It is a MASK
            if token_id == mask_id:
                decoded_tokens.append(f"{RED}█{RESET}")
            
            # 2. It is a NEWLY revealed token (compare with previous step)
            elif step > 0 and history[step-1][sample_idx][i] == mask_id:
                decoded_tokens.append(f"{GREEN}{token_str}{RESET}")
                
            # 3. It is a STABLE token (revealed previously)
            else:
                decoded_tokens.append(token_str)
        
        # Join and print
        full_text = "".join(decoded_tokens)
        print(f"{GRAY}Step {step:02d}:{RESET} {full_text}")

    print(f"{GRAY}{'='*60}{RESET}\n")
    
    
import time
import os
from IPython.display import clear_output, display

def animate_diffusion(pipeline_output, tokenizer, interval=0.2, sample_idx=0):
    """
    Animates the text generation process.
    Works in both Terminal and Jupyter Notebooks.
    """
    history = pipeline_output["history"]
    mask_id = tokenizer.mask_token_id
    
    # ANSI Colors
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    print(f"🎬 Starting Diffusion Animation ({len(history)} steps)...\n")
    time.sleep(1)

    for step, current_state in enumerate(history):
        # 1. Clear Screen
        # Check if running in Jupyter/Colab
        try:
            get_ipython()
            clear_output(wait=True)
        except NameError:
            # Running in standard terminal
            os.system('cls' if os.name == 'nt' else 'clear')
        
        # 2. Build the colored string
        current_ids = current_state[sample_idx]
        display_tokens = []
        
        for i, token_id in enumerate(current_ids):
            token_str = tokenizer.decode([token_id])
            
            # Logic: Determine color based on change from previous step
            if token_id == mask_id:
                # It's a mask -> Red Block
                display_tokens.append(f"{RED}█{RESET}")
            
            elif step > 0 and history[step-1][sample_idx][i] == mask_id:
                # It WAS a mask, now it's a word -> Green (Pop effect)
                display_tokens.append(f"{GREEN}{BOLD}{token_str}{RESET}")
                
            else:
                # Stable word -> Normal
                display_tokens.append(token_str)
                
        # 3. Print the frame
        full_text = "".join(display_tokens)
        print(f"\n{BOLD}Step {step}/{len(history)-1}{RESET}")
        print("-" * 40)
        print(full_text)
        print("-" * 40)
        
        # 4. Pause
        if step < len(history) - 1:
            time.sleep(interval)
        else:
            print(f"\n{GREEN}✨ Generation Complete!{RESET}")
            
            
def get_args(override_args: Optional[Dict[str, Any]] = None) -> tuple[ModelArguments, DataArguments, DiffusionTrainingArguments]:
    parser = HfArgumentParser((ModelArguments, DataArguments, DiffusionTrainingArguments)) # type: ignore
    
    if override_args is not None:
        # If called from sweep.py, we inject the dictionary as arguments
        # We need to convert dict to list of strings for parse_args_into_dataclasses if we were using sys.argv,
        # but HfArgumentParser has a parse_dict method!
        model_args, data_args, training_args = parser.parse_dict(override_args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Allow loading from a JSON config file: python train.py config.json
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Standard CLI parsing
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    return model_args, data_args, training_args