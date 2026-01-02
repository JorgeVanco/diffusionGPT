from datasets import load_dataset
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
    
    # Load Dataset
    raw_datasets = load_dataset(data_args.dataset_name)
    if data_args.max_train_samples:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
    # train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    # eval_dataset = load_dataset("roneneldan/TinyStories", split="validation[:200]")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    
    tokenizer.add_special_tokens(model_args.special_tokens)
    assert tokenizer.eos_token is not None, "The tokenizer must have an EOS token defined."
    assert tokenizer.mask_token is not None, "The tokenizer must have a MASK token defined."
    assert tokenizer.pad_token is not None, "The tokenizer must have a PAD token defined."

    # Data Processing (Packing / Grouping)
    def tokenize_function(examples) -> AutoTokenizer:
        outputs =  tokenizer(
            examples["text"], 
            padding=False,       # We pad dynamically in the collator
            truncation=True,     # Truncate to max model length
        )
        outputs["input_ids"] = [ids + [tokenizer.eos_token_id] for ids in outputs["input_ids"]]
        outputs["attention_mask"] = [am + [1] for am in outputs["attention_mask"]]
        return outputs
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Running tokenizer on dataset",
        )
    
    max_seq_length = model_args.max_seq_length
    def group_texts(examples):
        # Concatenate all texts in this batch
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the small remainder at the end of the batch
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
            
        # Split by chunks of max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    with training_args.main_process_first(desc="grouping texts"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=1000, # Higher batch size is faster for grouping
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )
    
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    def model_init():
        # Placeholder for model initialization
        from transformers import AutoModelForMaskedLM

        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        # print(config)
        # config = GPT2Config(
        #     vocab_size=len(tokenizer),
        #     n_positions=max_seq_length,  # Max sequence length
        #     n_embd=n_embd,       # Hidden dimension (Standard is 768)
        #     n_layer=n_layer,        # Number of layers (Standard is 12)
        #     n_head=n_head,         # Attention heads (Standard is 12)
        # )
        config.hidden_size = model_args.hidden_size
        config.num_hidden_layers = model_args.num_hidden_layers
        config.num_attention_heads = model_args.num_attention_heads
        config.intermediate_size = model_args.hidden_size * 4  # Typically 4x hidden size
        config.vocab_size = len(tokenizer)
        config.seq_length = max_seq_length
        config.mask_token_id = tokenizer.mask_token_id
        config.pad_token_id = tokenizer.pad_token_id
        # print(config)
        
        model = AutoModelForMaskedLM.from_config(config)
        # model.resize_token_embeddings(len(tokenizer))

        return model

    
    os.environ["WANDB_PROJECT"] = "text-diffusion"
    
    if training_args.auto_naming or not training_args.run_name:
        run_name = f"layers{model_args.num_hidden_layers}_embd{model_args.hidden_size}_seq{model_args.max_seq_length}_diff{training_args.num_diffusion_steps}_lr{training_args.learning_rate}_{datetime.now().strftime('%m%d_%H%M')}"
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
    seed_diffusion_curriculum_callback = SeedDiffusionCurriculumCallback()
    
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

    metrics = trainer.evaluate()
    
    return metrics["eval_loss"]

if __name__ == "__main__":
    main()