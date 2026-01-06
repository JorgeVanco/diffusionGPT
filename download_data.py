# Download once
import os
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from itertools import chain
from src.utils  import get_args
from src.data_utils import load_tokenizer, tokenize_and_pack


if __name__ == "__main__":
    model_args, data_args, training_args = get_args()
    
    tokenizer = load_tokenizer(model_args)
    
    # Load Dataset ------------------------------------------------
    dataset_dict = load_dataset(data_args.dataset_name, name=data_args.dataset_subset_name)
    
    lm_datasets = tokenize_and_pack(dataset_dict, tokenizer, model_args.max_seq_length)
    
    lm_datasets = lm_datasets.shuffle(seed=training_args.seed)
    
    lm_datasets.save_to_disk(data_args.load_from_disk)
    print(f"Datasets saved to {data_args.load_from_disk}")