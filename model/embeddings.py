from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math

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
    def __init__(self, embed_dim: int, t_min: float, t_max: float) -> None:
        super(TimeEmbeddings, self).__init__()
        
        self.t_min = t_min
        self.t_max = t_max
        
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.act = nn.GELU(approximate='tanh')
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        
        self.embed_dim = embed_dim
        
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def sample_t(self, n: int) -> Tensor:
        return torch.rand(n) * (self.t_max - self.t_min) + self.t_min

    def forward(self, time_steps: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time_steps, self.embed_dim)
        time_embeds = self.linear1(t_freq)
        time_embeds = self.act(time_embeds)
        time_embeds = self.linear2(time_embeds)
        time_embeds = self.act(time_embeds)
        
        return time_embeds