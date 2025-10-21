from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TimeEmbeddings

# Reference implementation of DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L19

@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    embed_dim: int = 256  # Changed from 1024 to match paper
    num_heads: int = 8
    num_layers: int = 8
    seq_len: int = 64


class DiffusionTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, 
                 num_layers: int, seq_len: int) -> None:
        super(DiffusionTransformer, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # Paper uses t_min=10, t_max=3000 for embed_dim=256
        self.time_embeddings = TimeEmbeddings(embed_dim, t_min=10.0, t_max=3000.0)
        
        self.rotary = Rotary(embed_dim // num_heads, seq_len)
        
        # Input: noisy_emb + clean_emb + prev_pred + mask = 4 * embed_dim
        self.embedding_proj = nn.Linear(embed_dim * 4, embed_dim)
        
        self.layers = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, self.rotary) for _ in range(num_layers)
        ])
        
        self.layernorm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )

    def forward(self, noisy_x: Tensor, clean_x: Tensor, prev_pred: Tensor,
                mask: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            noisy_x: [B, L, D] noisy embeddings (already scaled)
            clean_x: [B, L, D] clean conditioning embeddings
            prev_pred: [B, L, D] previous predictions (self-conditioning)
            mask: [B, L] or [B, L, 1] binary mask
            t: [B] timesteps
        """
        # Ensure mask is 3D
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        
        # Concatenate all inputs: noisy, clean, prev_pred, mask
        x = torch.cat([noisy_x, clean_x, prev_pred, mask.expand(-1, -1, self.embed_dim)], 
                     dim=-1)
        x = self.embedding_proj(x)
        
        # Time embedding
        c = self.time_embeddings(t)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x, c)
        
        # Final layer with adaptive layer norm
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.layernorm(x), shift, scale)
        
        # Project to vocabulary
        x = self.lm_head(x)
        
        return x


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        
        # Standard RoPE formulation
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor) -> Tensor:
        seq_len = x_BTHD.size(-3)
        cos = self.cos[:seq_len, None, :].to(x_BTHD.dtype)
        sin = self.sin[:seq_len, None, :].to(x_BTHD.dtype)
        
        # Split into pairs and apply rotation
        x1, x2 = x_BTHD[..., ::2], x_BTHD[..., 1::2]
        
        # Apply rotation
        rotated = torch.stack([
            x1 * cos[..., ::2] - x2 * sin[..., ::2],
            x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        ], dim=-1)
        
        return rotated.flatten(-2)
    
    
def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, rotary: Rotary) -> None:
        super(DiTBlock, self).__init__()
        self.rotary = rotary
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, rotary)
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
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention with gating
        x = x + gate_msa.unsqueeze(1) * self.attention(
            modulate(self.layernorm1(x), shift_msa, scale_msa))
        
        # FFN with gating
        x = x + gate_mlp.unsqueeze(1) * self.ffn(
            modulate(self.layernorm2(x), shift_mlp, scale_mlp))
        
        return x

    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rotary: Rotary) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rotary = rotary

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        q = self.rotary(q)
        k = self.rotary(k)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)
        
        return attn_output