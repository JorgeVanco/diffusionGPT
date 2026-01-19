import torch
import torch.nn.functional as F
from transformers import Trainer
import random

from src.utils import mask_input_ids_

class DiffusionTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.processing_class is None:
            raise ValueError("Tokenizer was not passed to the trainer!")
        
        labels = inputs.pop("labels")
        t = inputs.pop("t") # Timestep passed from collator
        
        if self.args.bf16:
            dtype = torch.bfloat16
            do_autocast = True
        elif self.args.fp16:
            dtype = torch.float16
            do_autocast = True
        else:
            dtype = torch.float32
            do_autocast = False
            
        device_type = "cpu" if self.args.use_cpu else "cuda"
        
        # Forward pass
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=do_autocast):
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Set logit that corresponds to the mask token to -inf
        logits[:, :, self.processing_class.mask_token_id] = torch.finfo(logits.dtype).min # type: ignore
        
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
        if self.args.time_loss_weighting: # type: ignore
            weights = (1.0 / (t + 1e-10)).view(-1, 1)  # Avoid division by zero
        else:
            weights = 1.0
        
        time_weight_mask = inputs.pop("time_weight_mask", None)
        if time_weight_mask is not None:
            weights = weights * time_weight_mask + 1.0 * (1.0 - time_weight_mask)
        
        # Mask out the ignored tokens from the average
        mask = (labels != -100).float()
        weighted_loss = (per_token_loss * mask * weights).sum() / (mask.sum() + 1e-10)
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss
    
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        logs["edit_stage_active"] = int(self.data_collator.edit_stage_active or 0)  # type: ignore
        logs["corruption_prob"] = self.data_collator.corruption_prob * logs["edit_stage_active"]  # type: ignore
        super().log(logs, start_time)
    
    
class DiscreteDiffusionCollator:
    def __init__(self, tokenizer, max_seq_length = None, corruption_prob=0.1, insertion_corruption=False) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.corruption_prob = corruption_prob
        self.insertion_corruption = insertion_corruption
        self.seed = 42
        self.generator = torch.Generator().manual_seed(self.seed)
        # Add a flag to control the curriculum
        self.edit_stage_active = False
        
        if insertion_corruption:
            if "<|delete|>" not in tokenizer.get_vocab():
                raise ValueError("Tokenizer must contain a <|delete|> token for insertion corruption.")
            self.delete_token_id = tokenizer.convert_tokens_to_ids("<|delete|>")

    def __call__(self, batch) -> dict[str, torch.Tensor]:
        insert_corruption_masks_cols = []
        insert_corruption_masks_rows = []
        for i, item in enumerate(batch): 
            insert_idx = None
            if self.edit_stage_active and (random.random() < self.corruption_prob):
                insert_idx = random.randint(0, len(item["input_ids"]) - 1)
                item["input_ids"].insert(insert_idx, self.delete_token_id)
                item["attention_mask"].insert(insert_idx, 1)
                if "assistant_masks" in item:
                    item["assistant_masks"].insert(insert_idx, 1)
                insert_corruption_masks_cols.append(insert_idx)
                insert_corruption_masks_rows.append(i)
            
            if self.max_seq_length is not None:
                # Truncate sequences longer than max_seq_length
                if insert_idx is not None and len(item['input_ids']) >= self.max_seq_length:
                    # If we truncated and the inserted index is out of bounds, adjust it
                    if len(item["input_ids"]) - self.max_seq_length < insert_idx:
                        insert_corruption_masks_cols[-1] = insert_idx - (len(item['input_ids']) - self.max_seq_length)
                    else:
                        insert_corruption_masks_cols.pop()
                        insert_corruption_masks_rows.pop()

                item['input_ids'] = item['input_ids'][-self.max_seq_length:]
                item['attention_mask'] = item['attention_mask'][-self.max_seq_length:]
                if "assistant_masks" in item:
                    item['assistant_masks'] = item['assistant_masks'][-self.max_seq_length:]
        
        # Get conversation mask if available
        if "assistant_masks" in batch[0]:
            assistant_masks_tensors = [torch.tensor(item.pop("assistant_masks")) for item in batch]
            assistant_masks = torch.nn.utils.rnn.pad_sequence(
                assistant_masks_tensors,
                batch_first=True,
                padding_value=1
            ).bool() # Pad with 1s (the model should not know when to finish so put ones until the end)
        else:
            assistant_masks = None
            
        
        batch_inputs = self.tokenizer.pad(
            batch, 
            padding=True, 
            return_tensors="pt"
        )
        
        input_ids = batch_inputs['input_ids']
        attention_mask = batch_inputs['attention_mask'].bool()
        
        if assistant_masks is not None:
            if assistant_masks.shape[1] < input_ids.shape[1]:
                # Pad assistant masks further if input_ids is longer
                diff = input_ids.shape[1] - assistant_masks.shape[1]
                assistant_masks = F.pad(assistant_masks, (0, diff), value=0)
            elif assistant_masks.shape[1] > input_ids.shape[1]:
                # Truncate if input_ids is shorter
                assistant_masks = assistant_masks[:, :input_ids.shape[1]]
                
            attention_mask = attention_mask & assistant_masks  # Ensure we only attend to assistant tokens if mask provided
        
        # Sample random timesteps for each example
        t = torch.rand((input_ids.size(0),), device=input_ids.device, generator=self.generator)
        
        noisy_inputs = input_ids.clone()
        
        mask_matrix = mask_input_ids_(
            noisy_inputs, 
            mask_token_id=self.tokenizer.mask_token_id, 
            mask_prob=t, 
            remasking_mask=assistant_masks,  # Only mask assistant tokens if mask provided
            generator=self.generator
        )
        labels = torch.full_like(input_ids, -100)
        labels[mask_matrix] = input_ids[mask_matrix]    # Calculate loss on MASKS (Standard MDLM)
        
        # Insert corruption masks
        insert_corruption_masks = torch.zeros_like(input_ids).bool()
        insert_corruption_masks[insert_corruption_masks_rows, insert_corruption_masks_cols] = True
        
        # Mask for time-weighting application
        # 1.0 = Apply w(t) (Standard Mask Loss)
        # 0.0 = Apply 1.0  (Edit/Reconstruction Loss)
        time_weight_mask = mask_matrix.float()
        
        if self.edit_stage_active:  # Apply Seed Diffusion Corruption Logic (Section 3.1 in https://arxiv.org/pdf/2508.02193)
            # Elegible for corruption
            eligible_for_corruption = ((~mask_matrix) & (attention_mask.bool())) | insert_corruption_masks
            
            corruption_matrix = (torch.rand(input_ids.shape, device=input_ids.device) < self.corruption_prob) | insert_corruption_masks
            corruption_mask = eligible_for_corruption & corruption_matrix
            
            # Generate random tokens for the corrupted positions
            random_tokens = torch.randint(0, len(self.tokenizer), input_ids.shape, device=input_ids.device)
            
            # Apply Random Swaps
            noisy_inputs[corruption_mask] = random_tokens[corruption_mask]
        
            labels[eligible_for_corruption] = input_ids[eligible_for_corruption] # Calculate loss on visible tokens (Seed Diffusion)
        
        return {
            'input_ids': noisy_inputs,
            'attention_mask': attention_mask,
            'labels': labels,
            't': t,
            'time_weight_mask': time_weight_mask
        }