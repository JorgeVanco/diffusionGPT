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
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    
    tokenizer.add_special_tokens(model_args.special_tokens)
    assert tokenizer.eos_token is not None, "The tokenizer must have an EOS token defined."
    assert tokenizer.mask_token is not None, "The tokenizer must have a MASK token defined."
    assert tokenizer.pad_token is not None, "The tokenizer must have a PAD token defined."

    # Load Dataset ------------------------------------------------
    max_train_samples = data_args.max_train_samples
    max_eval_samples = data_args.max_eval_samples
    dataset_name_clean = data_args.dataset_name.replace("/", "_")
    cache_name = f"cached_{dataset_name_clean}_{model_args.max_seq_length}_{model_args.tokenizer_name_or_path.replace('/', '_')}_{max_train_samples}_{max_eval_samples}"
    processed_data_path = os.path.join(training_args.output_dir, "data_cache", cache_name)
    if os.path.exists(processed_data_path) and not training_args.overwrite_output_dir:
        logging.info(f"✅ Loading pre-processed dataset from {processed_data_path}...")
        lm_datasets = load_from_disk(processed_data_path)
    else:
        logging.info(f"⚠️  Dataset cache not found at {processed_data_path}. Processing from scratch...")
        
        # Loading train
        dataset_stream = load_dataset(
            data_args.dataset_name, 
            name=data_args.dataset_subset_name if hasattr(data_args, "dataset_subset_name") else None,
            split="train",
            streaming=True
        )

        dataset_subset = Dataset.from_list(list(dataset_stream.take(max_train_samples))) # type: ignore
        dataset_subset = dataset_subset.select_columns(["text"]).shuffle(seed=training_args.seed)
        print("Total dataset size:", len(dataset_subset))
        
        available_splits = get_dataset_split_names(data_args.dataset_name, data_args.dataset_subset_name if hasattr(data_args, "dataset_subset_name") else None)
        val_split_name = next((s for s in available_splits if s in ["validation", "valid", "test"]), None)
        if val_split_name:
            val_stream = load_dataset(
                data_args.dataset_name, 
                name=data_args.dataset_subset_name if hasattr(data_args, "dataset_subset_name") else None,
                split=val_split_name,
                streaming=True
            )
            val_subset = Dataset.from_list(list(val_stream.take(max_eval_samples))) # type: ignore
            val_subset = val_subset.select_columns(["text"]).shuffle(seed=training_args.seed)
            raw_dataset = DatasetDict({
                "train": dataset_subset,
                "test": val_subset
            })
        else:
            # train/test split later
            raw_dataset = DatasetDict({
                "train": dataset_subset
            })

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
            tokenized_datasets = raw_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=os.cpu_count(),
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
                num_proc=os.cpu_count(),
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )
        
        if not val_split_name:
            lm_datasets = lm_datasets.train_test_split(test_size=max_eval_samples, seed=training_args.seed) # type: ignore
        
        if training_args.local_rank in [-1, 0]:
            logging.info(f"💾 Saving processed dataset to {processed_data_path}...")
            lm_datasets.save_to_disk(processed_data_path)
    
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["test"]

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
        config.pad_token_id = tokenizer.pad_token_id
        
        model = AutoModelForMaskedLM.from_config(config)
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

    metrics = trainer.evaluate()
    
    return metrics["eval_loss"]

if __name__ == "__main__":
    main()