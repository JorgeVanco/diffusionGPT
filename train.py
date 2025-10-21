from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from model.embeddings import TokenEmbeddings, EmbeddingsConfig
from model.transformer import DiffusionTransformer, TransformerConfig
from model.cdcd import CDCD
from model.loss import cdcd_loss
from model.scheduler import Scheduler


def main() -> None:
    # Initialize wandb
    wandb.init(
        project="text-diffusion",
        config={
            "learning_rate": 1e-4,  # Reduced from 5e-3
            "epochs": 100,
            "batch_size": 32,#128,  # Increased for better training
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 8,  # Increased from 6
            "seq_len": 64,
            "dataset": "TinyStories",
            "dataset_split": "train[:5%]",  # More data
            "t_min": 10.0,
            "t_max": 3000.0,
            "self_cond_prob": 0.5,
            "num_sampling_steps": 200,
        }
    )
    
    config = wandb.config
    
    # Load dataset and tokenizer
    dataset = load_dataset("roneneldan/TinyStories", split=f"{config.dataset_split}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=config.seq_len,
            return_tensors="pt"
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format(type='torch')
    dataloader = DataLoader(
        tokenized_datasets, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Model configuration
    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        seq_len=config.seq_len
    )
    
    model = CDCD(model_config)
    scheduler = Scheduler(tmin=config.t_min, tmax=config.t_max)
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Number of batches per epoch: {len(dataloader)}")
    
    # Optimizer with better settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs * len(dataloader),
        eta_min=config.learning_rate * 0.1
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training loop
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_num, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            
            optimizer.zero_grad()
            
            # Create random mask (50% prefix, 50% random as in paper)
            batch_size, seq_len = input_ids.shape
            mask = torch.ones(batch_size, seq_len, device=device)
            
            if torch.rand(1).item() < 0.5:
                # Prefix masking
                for i in range(batch_size):
                    prefix_len = torch.randint(0, seq_len // 2, (1,)).item()
                    mask[i, :prefix_len] = 0
            else:
                # Random masking
                for i in range(batch_size):
                    num_clean = torch.randint(0, seq_len // 2, (1,)).item()
                    clean_positions = torch.randperm(seq_len)[:num_clean]
                    mask[i, clean_positions] = 0
            
            # Forward pass with self-conditioning
            logits = model.train_forward(
                input_ids, 
                mask, 
                scheduler,
                self_cond_prob=config.self_cond_prob
            )
            
            # Compute loss only on masked positions
            loss = cdcd_loss(logits, input_ids, mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            lr_scheduler.step()
            
            # Logging
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            wandb.log({
                "loss": loss.item(),
                "epoch": epoch,
                "learning_rate": lr_scheduler.get_last_lr()[0]
            }, step=global_step)
            
            global_step += 1
            
            # Generate samples periodically
            if batch_num % 500 == 0 or batch_num == len(dataloader) - 1:
                model.eval()
                with torch.no_grad():
                    # Use first batch example for conditioning
                    test_input = input_ids[0:1]  # [1, L]
                    test_mask = mask[0:1]  # [1, L]
                    
                    # Generate 4 samples
                    generated_ids = model.generate(
                        batch_size=4,
                        steps=config.num_sampling_steps,
                        scheduler=scheduler,
                        input_ids=test_input.expand(4, -1),
                        mask=test_mask.expand(4, -1)
                    )
                    
                    # Create wandb table
                    gen_table = wandb.Table(columns=["sample_id", "conditioned_text", "generated_text", "full_text"])
                    
                    for i in range(generated_ids.size(0)):
                        # Get mask as 1D
                        mask_1d = test_mask[0].cpu()
                        
                        # Split into conditioned and generated parts
                        conditioned_indices = (mask_1d == 0).nonzero(as_tuple=True)[0]
                        generated_indices = (mask_1d == 1).nonzero(as_tuple=True)[0]
                        
                        if len(conditioned_indices) > 0:
                            conditioned_text = tokenizer.decode(
                                generated_ids[i][conditioned_indices], 
                                skip_special_tokens=True
                            )
                        else:
                            conditioned_text = ""
                        
                        if len(generated_indices) > 0:
                            generated_text = tokenizer.decode(
                                generated_ids[i][generated_indices], 
                                skip_special_tokens=True
                            )
                        else:
                            generated_text = ""
                        
                        full_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                        
                        print(f"Sample {i+1}: {conditioned_text}\033[92m{generated_text}\033[0m")
                        
                        gen_table.add_data(i+1, conditioned_text, generated_text, full_text)
                    
                    wandb.log({"generations": gen_table}, step=global_step)
                    
                model.train()
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch_loss": avg_epoch_loss}, step=global_step)
        
        # Save checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "loss": best_loss,
            }, "best_model_checkpoint.pth")
            print(f"Saved best model with loss: {best_loss:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "loss": avg_epoch_loss,
            }, f"model_checkpoint_epoch_{epoch}.pth")
    
    print("Training complete.")
    wandb.finish()

    
if __name__ == "__main__":
    main()