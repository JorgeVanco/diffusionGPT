import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizer
from itertools import chain
from src.utils  import get_args
from src.data_utils import load_tokenizer, tokenize_and_pack

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
        
        # Add special tokens to the vocabulary!
        special_tokens = ["<|im_start|>", "<|im_end|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
    return tokenizer

def process_smoltalk(examples, tokenizer):
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
    model_args, data_args, training_args = get_args()
    
    tokenizer = load_tokenizer(model_args)
    
    dataset = load_dataset("HuggingFaceTB/smol-smoltalk")
    
    tokenizer = setup_chat_format(tokenizer)
    
    dataset = dataset.map(
        lambda x: process_smoltalk(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count()
    )
    
    dataset = dataset.shuffle(seed=training_args.seed)
    
    dataset.save_to_disk(data_args.load_from_disk)
    print(f"Datasets saved to {data_args.load_from_disk}")