import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback

from src.utils import mask_input_ids_, _dispatch_table_logging

class TrainingInfoCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs) -> None:
        model = kwargs.get('model')
        train_dataloader = kwargs.get('train_dataloader')
        
        if model is None:
            raise ValueError("Model not found in on_train_begin kwargs.")
        
        # Calculate Parameter Count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate Dataset Size
        dataset_size = "Unknown"
        if train_dataloader:
            try:
                # Approximate if it's a generator, exact if it has __len__
                dataset_size = len(train_dataloader.dataset) 
            except:
                dataset_size = len(train_dataloader) * args.per_device_train_batch_size

        print("\n" + "="*40)
        print("🚀 TRAINING STARTED - CONFIGURATION")
        print("="*40)
        print(f"• Model Architecture:  {model.config.architectures[0] if model.config.architectures else 'Custom'}")
        print(f"• Total Parameters:    {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"• Trainable Params:    {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"• Max Seq Length:      {model.config.seq_length if hasattr(model.config, 'seq_length') else 'Unknown'}")
        print(f"• Dataset Size:        {dataset_size:,} examples")
        print(f"• Vocabulary Size:     {model.config.vocab_size}")
        print("-" * 40)
        print(f"• Total Epochs:        {args.num_train_epochs}")
        print(f"• Batch Size (Train):  {args.per_device_train_batch_size}")
        print(f"• Learning Rate:       {args.learning_rate}")
        print(f"• Total Steps:         {state.max_steps}")
        print(f"• Warmup Steps:        {args.warmup_steps}")
        print(f"• Logging to:          {args.output_dir}")
        print("="*40 + "\n")
        
class GenerativeEvalCallback(TrainerCallback):
    def __init__(self, test_prompts, tokenizer, pipeline_cls) -> None:
        self.prompts = test_prompts
        self.tokenizer = tokenizer
        self.pipeline_cls = pipeline_cls
        self.trainer: None | Trainer = None

    def on_evaluate(self, args, state, control, **kwargs) -> None:
        # The Trainer passes the 'model' to this method automatically
        model = kwargs.get("model")
        
        if model is None:
            print("⚠️ Warning: No model found in on_evaluate kwargs. Skipping generation.")
            return
        
        # Instantiate the pipeline dynamically using the current model
        pipe = self.pipeline_cls(
            model=model, 
            tokenizer=self.tokenizer,
            device=model.device 
        )
        
        steps = getattr(args, "num_diffusion_steps", 10)
        print(f"Running generative evaluation with {steps} steps...")
        pipe_outputs = pipe(self.prompts, num_steps=steps)
        outputs = [o["decoded_texts"][0] for o in pipe_outputs]
                
        if self.trainer:
            _dispatch_table_logging(self, content=outputs, step=state.global_step, trainer=self.trainer)
        else:
            print("⚠️ Warning: Trainer not set in GenerativeEvalCallback. Skipping logging.")
        
        print("\n" + "="*40)
        print("🧪 EVALUATION COMPLETED")
        print(f"• Step: {state.global_step}")
        print(f"• Eval Loss: {state.log_history[-1]['eval_loss']:.4f}")
        print("• Sample Outputs:")
        for i, output in enumerate(outputs):
            print(f"[{i}] {output}")
        print("="*40 + "\n")
        

class SeedDiffusionCurriculumCallback(TrainerCallback):
    def __init__(self) -> None:
        self.trainer: None | Trainer = None

    def on_step_begin(self, args, state, control, **kwargs) -> None:
        if self.trainer is None:
            raise ValueError("Trainer not set in SeedDiffusionCurriculumCallback.")
        
        # Check if we are past 80% of training (Two stage curriculum for Robust Diffusion Training - Section 3.1 in Seed Diffusion https://arxiv.org/pdf/2508.02193)
        threshold_step = state.max_steps * 0.8
        
        if state.global_step >= threshold_step:
            if hasattr(self.trainer.data_collator, "edit_stage_active"):
                if not self.trainer.data_collator.edit_stage_active:    # type: ignore
                    print(f"\n[Curriculum] Step {state.global_step}: Switching to Edit-Based Training Stage! 🔀")
                    self.trainer.data_collator.edit_stage_active = True # type: ignore


class DiffusionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.processing_class is None:
            raise ValueError("Tokenizer was not passed to the trainer!")
        
        labels = inputs.pop("labels")
        t = inputs.pop("t") # Timestep passed from collator
        
        # Forward pass
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
        # Real implementations use the exact derivative of the schedule
        weights = 1.0 / (t + 1e-6)  # Avoid division by zero
        
        # Mask out the ignored tokens from the average
        mask = (labels != -100).float()
        weighted_loss = (per_token_loss * mask * weights.unsqueeze(-1)).sum() / (mask.sum() + 1e-6)
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss
    
    
class DiscreteDiffusionCollator:
    def __init__(self, tokenizer, corruption_prob=0.1) -> None:
        self.tokenizer = tokenizer
        self.corruption_prob = corruption_prob
        self.seed = 42
        self.generator = torch.Generator().manual_seed(self.seed)
        # Add a flag to control the curriculum
        self.edit_stage_active = False

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
        
        mask_matrix = mask_input_ids_(
            noisy_inputs, 
            mask_token_id=self.tokenizer.mask_token_id, 
            mask_prob=t, 
            generator=self.generator
        )
        labels = torch.full_like(input_ids, -100)
        labels[mask_matrix] = input_ids[mask_matrix]    # Calculate loss on MASKS (Standard MDLM)
        
        if self.edit_stage_active:  # Apply Seed Diffusion Corruption Logic (Section 3.1 in https://arxiv.org/pdf/2508.02193)
            # Elegible for corruption
            eligible_for_corruption = (~mask_matrix) & (attention_mask.bool())
            
            corruption_matrix = torch.rand(input_ids.shape, device=input_ids.device) < self.corruption_prob
            corruption_mask = eligible_for_corruption & corruption_matrix
            
            # Generate random tokens for the corrupted positions
            random_tokens = torch.randint(0, len(self.tokenizer), input_ids.shape, device=input_ids.device)
            
            # Apply Random Swaps
            noisy_inputs[corruption_mask] = random_tokens[corruption_mask]
        
            # Ignore loss on unmasked tokens and padding tokens
            labels[corruption_mask] = input_ids[corruption_mask] # Calculate loss on CORRUPTIONS (Seed Diffusion)
        
        return {
            'input_ids': noisy_inputs,
            'attention_mask': attention_mask,
            'labels': labels,
            't': t
        }