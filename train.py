from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, GPT2Config
import os
from datetime import datetime

from src.trainer import DiffusionTrainer, DiscreteDiffusionCollator, TrainingInfoCallback, GenerativeEvalCallback
from src.pipeline import TextDiffusionPipeline
from src.DiffusionTrainingArguments import DiffusionTrainingArguments

def main() -> None:
    
    train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    eval_dataset = load_dataset("roneneldan/TinyStories", split="validation")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.mask_token = "<mask>"
    tokenizer.add_special_tokens({'mask_token': tokenizer.mask_token})
        
    max_seq_length = 128  # Define a max sequence length for TinyStories
        
    def tokenize_function(examples) -> AutoTokenizer:
        return tokenizer(
            examples["text"], 
            padding=False,       # We pad dynamically in the collator
            truncation=True,     # Truncate to max model length
            max_length=max_seq_length       # Set a reasonable length for TinyStories
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
    
    # Model parameters
    n_layer = 4
    n_embd = 256
    n_head = 4
    
    # Training parameters
    num_diffusion_steps = 50
    learning_rate = 2e-5
    batch_size = 16

    def model_init():
        # Placeholder for model initialization
        from transformers import AutoModelForCausalLM

        # config = AutoConfig.from_pretrained("gpt2")
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=max_seq_length,  # Max sequence length
            n_embd=n_embd,       # Hidden dimension (Standard is 768)
            n_layer=n_layer,        # Number of layers (Standard is 12)
            n_head=n_head,         # Attention heads (Standard is 12)
        )
        model = AutoModelForCausalLM.from_config(config)
        model.resize_token_embeddings(len(tokenizer))

        return model

    
    os.environ["WANDB_PROJECT"] = "text-diffusion"

    args = DiffusionTrainingArguments(
        # Diffusion specific
        num_diffusion_steps=num_diffusion_steps,
        # Output
        output_dir="output",
        # Optimization
        learning_rate=learning_rate,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={"num_decay_steps": 10000},
        warmup_steps=500,
        weight_decay=0.01,
        # Training
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # Evaluation & Saving
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        # Logging
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb",
        run_name=f"layers{n_layer}_embd{n_embd}_diff{num_diffusion_steps}_lr{learning_rate}_{datetime.now().strftime('%m%d_%H%M')}",
        # Misc
        remove_unused_columns=True,
    )
    
    data_collator = DiscreteDiffusionCollator(
        tokenizer=tokenizer
    )
    
    eval_callback = GenerativeEvalCallback(
        test_prompts=["Once upon a time", "There was a huge dragon"],
        tokenizer=tokenizer,
        pipeline_cls=TextDiffusionPipeline
    )
    
    trainer = DiffusionTrainer(
        model_init=model_init,
        args=args,
        train_dataset=tokenized_train,  # Placeholder for training dataset # type: ignore
        eval_dataset=tokenized_eval,    # Placeholder for evaluation dataset # type: ignore
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[TrainingInfoCallback(), eval_callback],
    )
    
    eval_callback.trainer = trainer  # Set trainer for logging purposes
    
    trainer.train()
    
if __name__ == "__main__":
    main()