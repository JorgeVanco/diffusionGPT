from transformers import TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, GPT2Config
import os
from datetime import datetime

from src.trainer import DiffusionTrainer, DiscreteDiffusionCollator, TrainingInfoCallback

def main() -> None:
    
    train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    eval_dataset = load_dataset("roneneldan/TinyStories", split="validation")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.mask_token = "<mask>"
    tokenizer.add_special_tokens({'mask_token': tokenizer.mask_token})
        
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding=False,       # We pad dynamically in the collator
            truncation=True,     # Truncate to max model length
            max_length=128       # Set a reasonable length for TinyStories
        )
    
    # Apply tokenization and remove the raw 'text' column
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )
    
    def model_init():
        # Placeholder for model initialization
        from transformers import AutoModelForCausalLM

        # config = AutoConfig.from_pretrained("gpt2")
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=512,  # Max sequence length
            n_embd=256,       # Hidden dimension (Standard is 768)
            n_layer=4,        # Number of layers (Standard is 12)
            n_head=4,         # Attention heads (Standard is 12)
        )
        model = AutoModelForCausalLM.from_config(config)
        model.resize_token_embeddings(len(tokenizer))

        return model

    
    os.environ["WANDB_PROJECT"] = "text-diffusion"

    args = TrainingArguments(
        output_dir="my_awesome_model",
        # Optimization
        learning_rate=2e-5,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={"num_decay_steps": 10000},
        warmup_steps=500,
        weight_decay=0.01,
        # Training
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # Evaluation & Saving
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # Logging
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb",
        run_name=f"text-diffusion-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        # Misc
        remove_unused_columns=True,
    )
    
    data_collator = DiscreteDiffusionCollator(
        tokenizer=tokenizer
    )
    
    trainer = DiffusionTrainer(
        model_init=model_init,
        args=args,
        train_dataset=tokenized_train,  # Placeholder for training dataset # type: ignore
        eval_dataset=tokenized_eval,    # Placeholder for evaluation dataset # type: ignore
        data_collator=data_collator,
        callbacks=[TrainingInfoCallback()],
    )
    trainer.train()
    
if __name__ == "__main__":
    main()