from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from model.embeddings import TokenEmbeddings, EmbeddingsConfig
from model.transformer import DiffusionTransformer, TransformerConfig
from model.loss import cdcd_loss


def main() -> None:
    # Load dataset and tokenizer
    dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch') #type:ignore
    dataloader = DataLoader(tokenized_datasets, batch_size=32, shuffle=True) #type:ignore
    
    # Model configuration
    config = TransformerConfig(
        vocab_size=EmbeddingsConfig.vocab_size,
        embed_dim=EmbeddingsConfig.embed_dim,
        num_heads=8,
        num_layers=6
    )
    
    # Initialize model, embeddings, optimizer, and loss function
    model = DiffusionTransformer(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers
    )
    token_embeddings = TokenEmbeddings(config.vocab_size, config.embed_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    for epoch in range(3):  # Train for 3 epochs
        for batch in dataloader:
            input_ids = torch.tensor(batch['input_ids'])
            
            optimizer.zero_grad()
            
            # Get token embeddings
            embeddings = token_embeddings(input_ids)
            
            # Add noise to embeddings
            t = model.time_embeddings.sample_t(embeddings.size(0))
            noisy_embeddings = embeddings + torch.rand_like(embeddings) * t.view(-1, 1, 1)
            
            
            # Forward pass through the transformer
            logits = model(noisy_embeddings, t)

            # Compute loss
            loss = cdcd_loss(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            

if __name__ == "__main__":
    main()