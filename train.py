from transformers import (
    AutoConfig, 
    set_seed,
)
import torch
from accelerate import Accelerator

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from src.data_utils import load_tokenizer, load_datasets
from src.utils import get_args
from src.trainer import DiffusionTrainer, DiscreteDiffusionCollator
from src.trainer_callbacks import TrainingInfoCallback, GenerativeEvalCallback, SeedDiffusionCurriculumCallback
from src.pipeline import TextDiffusionPipeline

torch.set_float32_matmul_precision('high')

def main(override_args: Optional[Dict[str, Any]] = None) -> float:
    
    model_args, data_args, training_args = get_args(override_args)

    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    set_seed(training_args.seed)
    
    tokenizer = load_tokenizer(model_args)
    
    accelerator = Accelerator()
    
    with accelerator.main_process_first():
        train_dataset, eval_dataset = load_datasets(model_args, data_args, training_args, tokenizer)

    # def model_init():
    # Model initialization
    from transformers import AutoModelForMaskedLM

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.hidden_size = model_args.hidden_size
    config.num_hidden_layers = model_args.num_hidden_layers
    config.num_attention_heads = model_args.num_attention_heads
    config.intermediate_size = model_args.hidden_size * 4
    config.vocab_size = len(tokenizer)
    config.seq_length = model_args.max_seq_length
    config.mask_token_id = tokenizer.mask_token_id
    config.pad_token_id = tokenizer.pad_token_id # type: ignore
    
    config.use_cache = False
    
    model = AutoModelForMaskedLM.from_config(config)

    if training_args.target_param_data_ratio is not None:
        total_params = sum(p.numel() for p in model.parameters())
        
        # 1. Calculate Total Tokens needed (e.g., Chinchilla: 20 * Params)
        total_tokens_needed = total_params * training_args.target_param_data_ratio
        
        # 2. Calculate Effective Batch Size (Batch x Grads x GPUs)
        effective_batch_size = (
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        )
        
        # 3. Calculate Tokens per Step (Batch x SeqLen)
        tokens_per_step = effective_batch_size * model_args.max_seq_length
        
        # 4. Set max_steps
        calculated_steps = int(total_tokens_needed / tokens_per_step)
        training_args.max_steps = calculated_steps
        
        print(f"\n🚀 DYNAMIC CONFIGURATION:")
        print(f"• Params: {total_params:,}")
        print(f"• Target Ratio: {training_args.target_param_data_ratio}")
        print(f"• Total Tokens Needed: {total_tokens_needed:,}")
        print(f"• Tokens Per Step: {tokens_per_step:,}")
        print(f"• Setting max_steps to: {calculated_steps:,}\n")
    
    os.environ["WANDB_PROJECT"] = "text-diffusion"
    
    if training_args.auto_naming or not training_args.run_name:
        run_name = f"layers{model_args.num_hidden_layers}_embd{model_args.hidden_size}_seq{model_args.max_seq_length}_diff{training_args.num_diffusion_steps}_lr{training_args.learning_rate}_{datetime.now().strftime('%m%d_%H%M')}"
        training_args.run_name = run_name
    else:
        run_name = training_args.run_name

    # Update output directory to include run name
    output_dir = os.path.join(training_args.output_dir, run_name)
    training_args.output_dir = output_dir

    data_collator = DiscreteDiffusionCollator(
        tokenizer=tokenizer,
        corruption_prob=training_args.corruption_prob
    )
    
    eval_callback = GenerativeEvalCallback(
        test_prompts=["Once upon a time", "There was a huge dragon"],
        tokenizer=tokenizer,
        pipeline_cls=TextDiffusionPipeline
    )
    seed_diffusion_curriculum_callback = SeedDiffusionCurriculumCallback(
        edit_stage_start=training_args.edit_stage_start,
        anneal_corruption=training_args.anneal_corruption
    )
    
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # Placeholder for training dataset # type: ignore
        eval_dataset=eval_dataset,    # Placeholder for evaluation dataset # type: ignore
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[TrainingInfoCallback(), eval_callback, seed_diffusion_curriculum_callback],
    )
    
    # Link trainer to callbacks that need it
    eval_callback.trainer = trainer  # Set trainer for logging purposes
    seed_diffusion_curriculum_callback.trainer = trainer  # Set trainer for curriculum callback
    
    # 8. Train & Evaluate
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()

    metrics = trainer.evaluate()
    
    return metrics["eval_loss"]

if __name__ == "__main__":
    main()