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
        print(f"• Max Seq Length:      {model.config.n_positions if hasattr(model.config, 'n_positions') else 'Unknown'}")
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
        outputs = pipe(self.prompts, num_steps=steps)
        
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
            print(f"\t[{i}] {output}")
        print("="*40 + "\n")


class DiffusionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.tokenizer is None:
            raise ValueError("Tokenizer was not passed to the trainer!")
        
        labels = inputs.pop("labels")
        t = inputs.pop("t") # Timestep passed from collator
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Set logit that corresponds to the mask token to -inf
        logits[:, :, self.tokenizer.mask_token_id] = torch.finfo(logits.dtype).min
        
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
        
        mask_matrix = mask_input_ids_(
            noisy_inputs, 
            mask_token_id=self.tokenizer.mask_token_id, 
            mask_prob=t, 
            generator=self.generator
        )
        
        # Ignore loss on unmasked tokens and padding tokens
        padding_mask = input_ids == self.tokenizer.pad_token_id
        input_ids[~mask_matrix | padding_mask] = -100
        
        return {
            'input_ids': noisy_inputs,
            'attention_mask': attention_mask,
            'labels': input_ids,
            't': t
        }