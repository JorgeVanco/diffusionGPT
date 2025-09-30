from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

@dataclass
class EmbeddingsConfig:
    vocab_size: int = 32000
    embed_dim: int = 256

class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super(TokenEmbeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
    @staticmethod
    def add_noise_embeddings(embeddings: Tensor, noise_level: float) -> Tensor:
        noise = torch.randn_like(embeddings) * noise_level
        return embeddings + noise
    
    def interpolate_embeddings(self, logits: Tensor) -> Tensor:
        return F.softmax(logits, dim=-1) @ self.token_embeddings.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        token_embeds = self.token_embeddings(input_ids)
        token_embeds = F.normalize(token_embeds, dim=-1) * (self.token_embeddings.embedding_dim ** 0.5)

        return token_embeds
    

class TimeEmbeddings(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super(TimeEmbeddings, self).__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.act = nn.GELU(approximate='tanh')
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)

    @staticmethod
    def sample_t(n: int) -> Tensor:
        return torch.rand(n)

    def forward(self, time_steps: Tensor) -> Tensor:
        time_embeds = self.linear1(time_steps)
        time_embeds = self.act(time_embeds)
        time_embeds = self.linear2(time_embeds)
        time_embeds = self.act(time_embeds)
        
        return time_embeds