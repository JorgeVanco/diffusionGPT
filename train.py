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
            "learning_rate": 5e-3,
            "epochs": 100,
            "batch_size": 32,
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 6,
            "seq_len": 64,
            "dataset": "TinyStories",
            "dataset_split": "train[:1%]"
        }
    )
    
    # Load dataset and tokenizer
    dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch') #type:ignore
    dataloader = DataLoader(tokenized_datasets, batch_size=32, shuffle=False) #type:ignore
    
    # Model configuration
    embeddings_config = EmbeddingsConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=256
    )
    
    config = TransformerConfig(
        vocab_size=embeddings_config.vocab_size,
        embed_dim=embeddings_config.embed_dim,
        num_heads=8,
        num_layers=6,
        seq_len=64
    )
    
    model = CDCD(config)
    scheduler = Scheduler(tmin=0.01,tmax=100)
    # Initialize model, embeddings, optimizer, and loss function
    # model = DiffusionTransformer(
    #     vocab_size=config.vocab_size,
    #     embed_dim=config.embed_dim,
    #     num_heads=config.num_heads,
    #     num_layers=config.num_layers,
    #     seq_len=config.seq_len
    # )
    print(f"Number of parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # token_embeddings = TokenEmbeddings(config.vocab_size, config.embed_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    
    print("Number of batches per epoch:", len(dataloader))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # token_embeddings = token_embeddings.to(device)
    
    # Training loop
    model.train()
    global_step = 0
    for epoch in range(100):  # Reduced to 100 epochs
        for batch_num, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            # print("Input IDs:", input_ids[0])
            optimizer.zero_grad()

            mask = torch.ones_like(input_ids).to(device)
            mask[:, :32] = 0
            logits = model.train_forward(input_ids, mask, scheduler)

            # Compute loss
            loss = cdcd_loss(logits, input_ids)
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
            # Log loss to wandb
            wandb.log({"loss": loss.item(), "epoch": epoch}, step=global_step)
            global_step += 1
            
            if batch_num % 500 == 0 or batch_num == len(dataloader) - 1:
                model.eval()
                with torch.no_grad():
                    predicted_ids = model.generate(batch_size=4, steps=1000, scheduler=scheduler, input_ids=input_ids[0].unsqueeze(0).expand(4, -1), mask=mask[0].unsqueeze(0).expand(4, -1))
                    print("Generated logits shape:", predicted_ids.shape)
                    # logits = model.lm_head(generated_logits)
                    # predicted_ids = torch.argmax(logits, dim=-1)
                    
                    # Create wandb table for generated samples
                    generation_table = wandb.Table(columns=["sample_id", "conditioned_text", "generated_text", "full_text"])
                    
                    for i in range(predicted_ids.size(0)):
                        # Split into conditioned and generated parts based on mask
                        conditioned_part = predicted_ids[i][mask[0] == 0]
                        generated_part = predicted_ids[i][mask[0] == 1]
                        
                        conditioned_text = tokenizer.decode(conditioned_part, skip_special_tokens=True)
                        generated_text = tokenizer.decode(generated_part, skip_special_tokens=True)
                        full_text = conditioned_text + generated_text
                        
                        print(f"Generated Text {i+1}: {conditioned_text}\033[92m{generated_text}\033[0m")
                        
                        # Add to wandb table
                        generation_table.add_data(i+1, conditioned_text, generated_text, full_text)
                    
                    # Log the generation table to wandb
                    wandb.log({"generations": generation_table}, step=global_step)
                    
                model.train()

                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


        torch.save({
            "model_state_dict": model.state_dict(),
            # "token_embeddings_state_dict": token_embeddings.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, "model_checkpoint.pth")
    
    print("Training complete.")
    wandb.finish()

    
if __name__ == "__main__":
    main()