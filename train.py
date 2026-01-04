from datasets import load_dataset, load_from_disk, Dataset, get_dataset_split_names, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    HfArgumentParser, 
    set_seed,
)
import torch

import os
import sys
import logging
from datetime import datetime
from itertools import chain
from typing import Optional, Dict, Any

from src.data_utils import load_tokenizer, load_datasets
from src.trainer import DiffusionTrainer, DiscreteDiffusionCollator
from src.trainer_callbacks import TrainingInfoCallback, GenerativeEvalCallback, SeedDiffusionCurriculumCallback
from src.pipeline import TextDiffusionPipeline
from src.argument_classes import DiffusionTrainingArguments, ModelArguments, DataArguments

torch.set_float32_matmul_precision('high')

def main(override_args: Optional[Dict[str, Any]] = None) -> float:
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
        
    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    set_seed(training_args.seed)
    
    tokenizer = load_tokenizer(model_args)
    
    # train_dataset, eval_dataset = load_datasets(model_args, data_args, training_args, tokenizer)

    def model_init():
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
        return model

    
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
        model_init=model_init,
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
        trainer.train()
        trainer.save_model()

    metrics = trainer.evaluate()
    
    return metrics["eval_loss"]

if __name__ == "__main__":
    main()