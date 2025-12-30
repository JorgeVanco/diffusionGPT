from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, GPT2Config
import os
from datetime import datetime
from itertools import chain

from src.trainer import DiffusionTrainer, DiscreteDiffusionCollator
from src.trainer_callbacks import TrainingInfoCallback, GenerativeEvalCallback, SeedDiffusionCurriculumCallback
from src.pipeline import TextDiffusionPipeline
from src.DiffusionTrainingArguments import DiffusionTrainingArguments

def main() -> None:
    
    train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    eval_dataset = load_dataset("roneneldan/TinyStories", split="validation")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
    tokenizer.mask_token = "<mask>"
    tokenizer.add_special_tokens({'mask_token': tokenizer.mask_token})
        
    max_seq_length = 128  # Define a max sequence length for TinyStories
        
    def tokenize_function(examples) -> AutoTokenizer:
        outputs =  tokenizer(
            examples["text"], 
            padding=False,       # We pad dynamically in the collator
            truncation=True,     # Truncate to max model length
        )
        outputs["input_ids"] = [ids + [tokenizer.eos_token_id] for ids in outputs["input_ids"]]
        outputs["attention_mask"] = [am + [1] for am in outputs["attention_mask"]]
        return outputs
    
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
    
    packed_train = tokenized_train.map(group_texts, batched=True)
    packed_eval = tokenized_eval.map(group_texts, batched=True)
    
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
        from transformers import AutoModelForMaskedLM

        config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
        # print(config)
        # config = GPT2Config(
        #     vocab_size=len(tokenizer),
        #     n_positions=max_seq_length,  # Max sequence length
        #     n_embd=n_embd,       # Hidden dimension (Standard is 768)
        #     n_layer=n_layer,        # Number of layers (Standard is 12)
        #     n_head=n_head,         # Attention heads (Standard is 12)
        # )
        config.hidden_size = n_embd
        config.num_hidden_layers = n_layer
        config.num_attention_heads = n_head
        config.intermediate_size = n_embd * 4  # Typically 4x hidden size
        config.vocab_size = len(tokenizer)
        config.seq_length = max_seq_length
        config.mask_token_id = tokenizer.mask_token_id
        config.pad_token_id = tokenizer.pad_token_id
        # print(config)
        
        model = AutoModelForMaskedLM.from_config(config)
        # model.resize_token_embeddings(len(tokenizer))

        return model

    
    os.environ["WANDB_PROJECT"] = "text-diffusion"
    
    run_name = f"layers{n_layer}_embd{n_embd}_seq{max_seq_length}_diff{num_diffusion_steps}_lr{learning_rate}_{datetime.now().strftime('%m%d_%H%M')}"

    args = DiffusionTrainingArguments(
        # Diffusion specific
        num_diffusion_steps=num_diffusion_steps,
        # Output
        output_dir=f"output/{run_name}",
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
        run_name=run_name,
        # Misc
        remove_unused_columns=True,
    )
    
    data_collator = DiscreteDiffusionCollator(
        tokenizer=tokenizer,
        corruption_prob=args.corruption_prob
    )
    
    eval_callback = GenerativeEvalCallback(
        test_prompts=["Once upon a time", "There was a huge dragon"],
        tokenizer=tokenizer,
        pipeline_cls=TextDiffusionPipeline
    )
    seed_diffusion_curriculum_callback = SeedDiffusionCurriculumCallback()
    
    trainer = DiffusionTrainer(
        model_init=model_init,
        args=args,
        train_dataset=packed_train,  # Placeholder for training dataset # type: ignore
        eval_dataset=packed_eval,    # Placeholder for evaluation dataset # type: ignore
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[TrainingInfoCallback(), eval_callback, seed_diffusion_curriculum_callback],
    )
    
    eval_callback.trainer = trainer  # Set trainer for logging purposes
    seed_diffusion_curriculum_callback.trainer = trainer  # Set trainer for curriculum callback
    
    trainer.train()
    
if __name__ == "__main__":
    main()