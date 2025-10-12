import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from model.embeddings import TokenEmbeddings
from model.scheduler import Scheduler
from model.transformer import DiffusionTransformer
from model.embeddings import TokenEmbeddings

def score_interpolation(logits: Tensor, input_embedding: Tensor, t: Tensor, embeddings: TokenEmbeddings) -> Tensor:
    mean_embedding = embeddings.interpolate_embeddings(logits)
    score = (mean_embedding - input_embedding) / t**2
    return score


def diffusion_step(logits: Tensor, input_embedding: Tensor, t: Tensor, embeddings: TokenEmbeddings, lr: float) -> Tensor:
    score = score_interpolation(logits, input_embedding, t, embeddings)
    updated_embedding = input_embedding + lr * score
    return updated_embedding


class CDCD(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = DiffusionTransformer(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            seq_len=config.seq_len
        )
        self.seq_len = config.seq_len
        self.token_embeddings = TokenEmbeddings(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim
        )
        self.time_embeddings = self.model.time_embeddings

    def train_forward(self, input_ids: Tensor, mask: Tensor, scheduler: Scheduler) -> Tensor:
        batch_size = input_ids.size(0)
        embeddings = self.token_embeddings(input_ids)
        t = scheduler.sample((batch_size,)).to(embeddings.device) #type:ignore
        mask = mask.unsqueeze(-1).expand(embeddings.shape).to(embeddings.device)
        noisy_embeddings = embeddings + torch.randn_like(embeddings) * t.view(-1, 1, 1) * mask
        clean_embeddings = embeddings * (1 - mask)
        
        logits = self.model(noisy_embeddings, clean_embeddings, mask, t)
        return logits
    
    @torch.no_grad()
    def denoise(self, noised_embeddings: Tensor, clean_embeddings: Tensor, mask: Tensor, t: float, steps: int, scheduler: Scheduler) -> Tensor:
        device = noised_embeddings.device
        batch_size = noised_embeddings.size(0)
        timesteps = scheduler.make_timesteps(steps, tmax=t)
        x_i = torch.empty_like(noised_embeddings)
        
        for i in range(steps-1):
            t_i = timesteps[i].to(device)
            logits = self.model(noised_embeddings, clean_embeddings, mask, torch.full((batch_size,), t_i.item(), device=device)) #self.model.forward_embeddings(x_i if i > 0 else noised_embeddings, t_i, self.token_embeddings)
            # lr = scheduler.lr(t_i).view(-1, 1, 1)
            # x_i = diffusion_step(logits, noised_embeddings, t_i.view(-1, 1, 1), self.token_embeddings, lr)
            x_i = self.token_embeddings.interpolate_embeddings(logits)
            
            derivative = (noised_embeddings - x_i) / t_i
            delta_t = timesteps[i+1] - t_i
            noised_embeddings = noised_embeddings + derivative * delta_t
        return self.model(noised_embeddings, clean_embeddings, mask, torch.full((batch_size,), timesteps[-1].item(), device=device))

    def generate(self, batch_size: int, steps: int, scheduler: Scheduler, input_ids: Tensor | None = None, mask: Tensor | None = None) -> Tensor:

        if mask is None:
            mask = torch.ones((batch_size, self.seq_len, self.token_embeddings.token_embeddings.embedding_dim), device="cuda")
            noised_embeddings = torch.randn(batch_size, self.seq_len, self.token_embeddings.token_embeddings.embedding_dim, device="cuda") * scheduler.tmax # Example shape (batch_size, seq_len, vocab_size)
            clean_embeddings = torch.zeros_like(noised_embeddings)
        else:
            if mask.shape != (batch_size, self.seq_len, self.token_embeddings.token_embeddings.embedding_dim):
                mask = mask.unsqueeze(-1).expand(batch_size, self.seq_len, self.token_embeddings.token_embeddings.embedding_dim)
            if input_ids is None:
                raise ValueError("input_ids must be provided if mask is provided")
            embeddings = self.token_embeddings(input_ids.to("cuda"))
            clean_embeddings = embeddings * (1 - mask.to("cuda"))
            noised_embeddings = embeddings + torch.randn_like(embeddings) * scheduler.tmax * mask.to("cuda")
        
        logits = self.denoise(noised_embeddings, clean_embeddings, mask, scheduler.tmax, steps, scheduler)
        probs = F.softmax(logits, dim=-1)
        generated_ids = torch.argmax(probs, dim=-1)
        return generated_ids * mask[:,:,0] + (1 - mask[:,:,0]) * input_ids