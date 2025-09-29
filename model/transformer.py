from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    embed_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 8

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int) -> None:
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.layernorm(x)
        x = self.lm_head(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention(self.layernorm1(x)) + x
        x = self.ffn(self.layernorm2(x)) + x
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
    model = Transformer(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers
    )
    x = torch.randn(1, 16, config.embed_dim)
    out = model(x)
    print(out.shape)  # Expected output shape: (1, 16, vocab_size)