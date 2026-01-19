import os
import argparse
from datasets import concatenate_datasets, Dataset
from src.utils import get_args
from src.data_utils import load_tokenizer
from tasks import TASK_REGISTRY

def setup_chat_format(tokenizer):
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{% if message['role'] == 'assistant' %}"
            "{% generation %}"
            "{{ message['content'] }}"
            "<|im_end|>"
            "{% endgeneration %}"
            "{% else %}"
            "{{ message['content'] }}"
            "<|im_end|>"
            "{% endif %}"
            "\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        )
        
        # Add special tokens to the vocabulary
        special_tokens = ["<|im_start|>", "<|im_end|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
    return tokenizer

def process_batch(examples, tokenizer):
    processed_examples = {
        "input_ids": [],
        "attention_mask": [],
        "assistant_masks": []
    }
    
    for conversation in examples["messages"]:
        for i, message in enumerate(conversation):
            if message["role"] == "assistant":
                conversation_slice = conversation[:i+1]
                tokenized = tokenizer.apply_chat_template(
                    conversation_slice,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                    padding=False,
                    truncation=False
                )
                processed_examples["input_ids"].append(tokenized["input_ids"])
                processed_examples["attention_mask"].append(tokenized["attention_mask"])
                processed_examples["assistant_masks"].append(tokenized["assistant_masks"])

    return processed_examples

if __name__ == "__main__":
    task_names = "everyday,smoltalk,nemotron"
    
    model_args, data_args, training_args = get_args()
    
    tokenizer = load_tokenizer(model_args)
    tokenizer = setup_chat_format(tokenizer)
    
    # 1. Load and Mix Datasets
    task_names = task_names.split(",")
    raw_datasets = []
    
    print(f"Loading tasks: {task_names}")
    for name in task_names:
        name = name.strip()
        if name in TASK_REGISTRY:
            task = TASK_REGISTRY[name](seed=training_args.seed)
            ds = task.load_dataset()
            
            # Ensure only necessary columns are kept to avoid concatenation errors
            ds = ds.select_columns(["messages"])
            raw_datasets.append(ds)
            print(f"Loaded {name}: {len(ds)} samples")
        else:
            print(f"Warning: Task '{name}' not found in registry.")

    if not raw_datasets:
        raise ValueError("No valid datasets loaded.")

    # Concatenate all loaded datasets
    combined_dataset = concatenate_datasets(raw_datasets)
    
    # Shuffle mixed dataset
    combined_dataset = combined_dataset.shuffle(seed=training_args.seed)
    print(f"Total samples after mixing: {len(combined_dataset)}")

    # 2. Process (Tokenize)
    processed_dataset = combined_dataset.map(
        lambda x: process_batch(x, tokenizer),
        batched=True,
        remove_columns=combined_dataset.column_names,
        num_proc=os.cpu_count()
    )
    
    # 3. Save
    processed_dataset.save_to_disk(data_args.load_from_disk)
    print(f"Datasets saved to {data_args.load_from_disk}")