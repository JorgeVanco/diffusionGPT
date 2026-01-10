from transformers import (
    AutoModelForMaskedLM, 
    set_seed,
)
import torch
from accelerate import Accelerator

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from preprocess_chat_dataset import setup_chat_format
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
    tokenizer = setup_chat_format(tokenizer)
    
    accelerator = Accelerator()
    
    with accelerator.main_process_first():
        train_dataset, eval_dataset = load_datasets(model_args, data_args, training_args, tokenizer)

    # Model initialization
    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer)) # Resize embeddings in case new special tokens were added
    
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
        corruption_prob=training_args.corruption_prob,
        max_seq_length=model_args.max_seq_length
    )
    
    eval_callback = GenerativeEvalCallback(
        test_prompts=["<|im_start|>user\nI am concerned my lack of a college degree in my field hurts my long term career prospects despite my relevant work experience and skills. Is this a valid concern?<|im_end|>\n<|im_start|>assistant\n", "<|im_start|>system\nYou're an AI assistant for text re-writing. Rewrite the input text to make it more friendly and approachable while maintaining its main points.<|im_end|>\n<|im_start|>user\nDr. Thompson,\n\nI've reviewed the latest draft of the Listeria campaign, and I must say I'm disappointed. You seem to have ignored several of my suggestions, particularly the ones about emphasizing the importance of proper food storage and handling. I understand you might have your own ideas, but this is a critical issue, and we need to ensure the public is fully informed.\n\nIf this keeps up, I'll have to escalate this to our department head.\n\nEmily Jenkins, Ph.D.\nPublic Health Professional, Food Safety Specialist<|im_end|>\n<|im_start|>assistant\n"],
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