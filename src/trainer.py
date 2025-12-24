import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback

class TrainingInfoCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs) -> None:
        print("\n" + "="*40)
        print("🚀 TRAINING STARTED - CONFIGURATION")
        print("="*40)
        print(f"• Total Epochs:      {args.num_train_epochs}")
        print(f"• Batch Size (Train):{args.per_device_train_batch_size}")
        print(f"• Learning Rate:     {args.learning_rate}")
        print(f"• Total Steps:       {state.max_steps}")
        print(f"• Warmup Steps:      {args.warmup_steps}")
        print(f"• Logging to:        {args.output_dir}")
        print("="*40 + "\n")

class DiffusionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        labels = inputs.pop("labels")
        t = inputs.pop("t") # Timestep passed from collator
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Flatten for CrossEntropy
        # logits: (B, L, V), labels: (B, L)
        # Only compute loss on masked tokens (labels!= -100)
        # Note: CrossEntropyLoss automatically ignores -100
        per_token_loss = F.cross_entropy(logits.view(-1, self.model.config.vocab_size),  # type: ignore
                                  labels.view(-1), reduction='none')
        
        # Reshape to (B, L) to apply time weighting
        per_token_loss = per_token_loss.view(labels.shape)
        
        # Apply Loss Weighting (MDLM formulation)
        # Assume linear schedule for example: w(t) = 1/t
        # Real implementations use the exact derivative of the schedule
        weights = 1.0 / (t + 1e-6)  # Avoid division by zero
        
        # Mask out the ignored tokens from the average
        mask = (labels != -100).float()
        weighted_loss = (per_token_loss * mask * weights.unsqueeze(-1)).sum() / (mask.sum() + 1e-6)
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss
    
    
class DiscreteDiffusionCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.seed = 42
        self.generator = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)

    def __call__(self, batch) -> dict[str, torch.Tensor]:
        batch_inputs = self.tokenizer.pad(
            batch, 
            padding=True, 
            return_tensors="pt"
        )
        
        input_ids = batch_inputs['input_ids']
        attention_mask = batch_inputs['attention_mask']
        
        # Sample random timesteps for each example
        t = torch.rand((input_ids.size(0),), device=input_ids.device, generator=self.generator)
        
        
        noisy_inputs = input_ids.clone()
        
        # For each position, with probability t, replace with mask token id
        prob_matrix = torch.rand(input_ids.shape, device=input_ids.device, generator=self.generator)
        mask_token_id = self.tokenizer.mask_token_id
        mask_matrix = prob_matrix < t.view(-1, 1)
        noisy_inputs[mask_matrix] = mask_token_id
        
        return {
            'input_ids': noisy_inputs,
            'attention_mask': attention_mask,
            'labels': input_ids,
            't': t
        }