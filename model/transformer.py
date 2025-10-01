from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TimeEmbeddings, TokenEmbeddings

# Reference implementation of DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L19

@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    embed_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 8

class DiffusionTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int) -> None:
        super(DiffusionTransformer, self).__init__()

        # self.token_embeddings = TokenEmbeddings(vocab_size, embed_dim)
        self.time_embeddings = TimeEmbeddings(embed_dim, t_min=1.0, t_max=100.0)

        self.layers = nn.ModuleList([
            DiTBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        
        # x = self.token_embeddings(x)
        c = self.time_embeddings(t)
        
        for layer in self.layers:
            x = layer(x, c)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.layernorm(x), shift, scale)
        x = self.lm_head(x)
        return x

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super(DiTBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True)
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attention(modulate(self.layernorm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.ffn(modulate(self.layernorm2(x), shift_mlp, scale_mlp))
        return x

def apply_rope(x: Tensor, seq_len: int, dim: int) -> Tensor:
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
    pe = torch.zeros(seq_len, 1, dim)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    x = x + pe
    return x
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q, k = apply_rope(q, x.size(0), q.size(-1)), apply_rope(k, x.size(0), k.size(-1))
        
        q = q.view(x.size(0), -1, self.num_heads, q.size(-1) // self.num_heads).transpose(1, 2)
        k = k.view(x.size(0), -1, self.num_heads, k.size(-1) // self.num_heads).transpose(1, 2)
        v = v.view(x.size(0), -1, self.num_heads, v.size(-1) // self.num_heads).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), -1, self.num_heads * (q.size(-1)))
        attn_output = self.out_proj(attn_output)
        return attn_output
        
        
class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()

        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor) -> Tensor:
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)
    

if __name__ == "__main__":
    config = TransformerConfig()
    model = DiffusionTransformer(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers
    )
    b = 4
    indices = torch.randint(0, config.vocab_size, (b, 16))
    token_embeddings = TokenEmbeddings(config.vocab_size, config.embed_dim)
    x = token_embeddings(indices)
    t = torch.randn(b)
    out = model(x, t)
    print(out.shape)  # Expected output shape: (b, 16, vocab_size)