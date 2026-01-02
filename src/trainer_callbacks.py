from transformers import Trainer, TrainerCallback

from src.utils import _dispatch_table_logging


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
    def __init__(self, edit_stage_start: float = 0.8, anneal_corruption: bool = True) -> None:
        self.trainer: None | Trainer = None
        self.edit_stage_start: float = edit_stage_start
        self.anneal_corruption: bool = anneal_corruption

    def on_step_begin(self, args, state, control, **kwargs) -> None:
        if self.trainer is None:
            raise ValueError("Trainer not set in SeedDiffusionCurriculumCallback.")
        
        # Check if we are past 80% of training (Two stage curriculum for Robust Diffusion Training - Section 3.1 in Seed Diffusion https://arxiv.org/pdf/2508.02193)
        threshold_step = state.max_steps * self.edit_stage_start
        
        if state.global_step >= threshold_step:
            if hasattr(self.trainer.data_collator, "edit_stage_active"):
                if not self.trainer.data_collator.edit_stage_active:    # type: ignore
                    print(f"\n[Curriculum] Step {state.global_step}: Switching to Edit-Based Training Stage! 🔀")
                    self.trainer.data_collator.edit_stage_active = True # type: ignore
            
            target_corruption = getattr(args, "corruption_prob", 0.1)
            if self.anneal_corruption:
                progress = state.global_step / state.max_steps
                edit_progress = (progress - self.edit_stage_start) / ((1.0 + self.edit_stage_start) / 2 - self.edit_stage_start)
                
                
                current_prob = min(target_corruption * edit_progress, target_corruption)
                self.trainer.data_collator.corruption_prob = current_prob # type: ignore
            else:
                self.trainer.data_collator.corruption_prob = target_corruption # type: ignore