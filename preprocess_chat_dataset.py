import os
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from itertools import chain
from src.utils  import get_args
from src.data_utils import load_tokenizer, tokenize_and_pack

def setup_chat_format(tokenizer):
    # If your tokenizer doesn't have a chat_template, define a simple standard one (ChatML-like)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
        
        # Make sure you add these special tokens to the vocabulary!
        special_tokens = ["<|im_start|>", "<|im_end|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
    return tokenizer

def process_smoltalk(examples, tokenizer):
    # Convert list of dicts -> Single String
    # e.g., "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>\n"
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) 
        for messages in examples['messages']
    ]
    return {"text": texts}

if __name__ == "__main__":
    model_args, data_args, training_args = get_args()
    
    tokenizer = load_tokenizer(model_args)
    
    dataset = load_dataset("HuggingFaceTB/smoltalk", "all")
    
    tokenizer = setup_chat_format(tokenizer)
    
    dataset = dataset.map(lambda x: process_smoltalk(x, tokenizer), batched=True, remove_columns=dataset['train'].column_names)
    
    dataset = tokenize_and_pack(dataset, tokenizer, model_args.max_seq_length)
    
    lm_datasets = dataset.shuffle(seed=training_args.seed)
    
    lm_datasets.save_to_disk(data_args.load_from_disk)
    print(f"Datasets saved to {data_args.load_from_disk}")