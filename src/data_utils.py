from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk, get_dataset_split_names
from itertools import chain
import logging
import os

def load_tokenizer(model_args) -> PreTrainedTokenizer:
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    
    tokenizer.add_special_tokens(model_args.special_tokens)
    assert tokenizer.eos_token is not None, "The tokenizer must have an EOS token defined."
    assert tokenizer.mask_token is not None, "The tokenizer must have a MASK token defined."
    assert tokenizer.pad_token is not None, "The tokenizer must have a PAD token defined."
    
    return tokenizer


def load_datasets(model_args, data_args, training_args, tokenizer):
    # Load Dataset ------------------------------------------------
    target_train_samples = data_args.max_train_samples
    target_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples else 1000
    
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
    
    # ----------------------------------------------------------------------------
    # PATH A: STATIC (TinyStories)
    # ----------------------------------------------------------------------------
    if not data_args.streaming:
        logging.info(f"📚 Loading {data_args.dataset_name} (Static)...")

        # Load & Split
        subset = data_args.dataset_subset_name
        raw_datasets = load_dataset(data_args.dataset_name, name=subset)
        
        # Check for validation split
        try:
            splits = get_dataset_split_names(data_args.dataset_name, subset)
            has_val = any(s in ["validation", "valid", "test"] for s in splits)
        except:
            has_val = False

        if has_val:
            val_split = next(s for s in splits if s in ["validation", "valid", "test"])
            train_ds = raw_datasets["train"]
            test_ds = raw_datasets[val_split]
        else:
            logging.info("ℹ️ No validation split found. Splitting 'train' manually.")

            split_ds = raw_datasets["train"].train_test_split(test_size=target_eval_samples, seed=training_args.seed)
            train_ds = split_ds["train"]
            test_ds = split_ds["test"]

        # Truncate (Optional)
        if target_train_samples is not None:
            train_ds = train_ds.select(range(min(len(train_ds), target_train_samples)))
        test_ds = test_ds.select(range(min(len(test_ds), target_eval_samples)))

        # Combine for uniform processing
        dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})
        dataset_dict = dataset_dict.select_columns(["text"])
        
        # Process
        with training_args.main_process_first(desc="tokenizing"):
            tokenized = dataset_dict.map(tokenize_function, batched=True, num_proc=os.cpu_count())
        
        with training_args.main_process_first(desc="grouping"):
            lm_datasets = tokenized.map(group_texts, batched=True, batch_size=1000, num_proc=os.cpu_count())

        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["test"]

    # ----------------------------------------------------------------------------
    # PATH B: STREAMING (FineWeb)
    # ----------------------------------------------------------------------------
    else:
        logging.info(f"🌊 Streaming {data_args.dataset_name}...")
        
        raw_stream = load_dataset(
            data_args.dataset_name, 
            name=data_args.dataset_subset_name, 
            split="train", 
            streaming=True
        )

        # Split & Shuffle
        eval_stream = raw_stream.take(target_eval_samples)
        train_stream = raw_stream.skip(target_eval_samples)
        train_stream = train_stream.shuffle(seed=training_args.seed, buffer_size=10_000)
        
        # Lazy Processing
        tokenized_train = train_stream.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_eval = eval_stream.map(tokenize_function, batched=True, remove_columns=["text"])
        
        train_dataset = tokenized_train.map(group_texts, batched=True, batch_size=1000)
        eval_dataset = tokenized_eval.map(group_texts, batched=True, batch_size=1000)
        
        if target_train_samples is not None:
            train_dataset = train_dataset.take(target_train_samples)
    
    return train_dataset, eval_dataset