import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from model.embeddings import TokenEmbeddings
from model.scheduler import Scheduler
from model.transformer import DiffusionTransformer


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

    def train_forward(self, input_ids: Tensor, mask: Tensor, scheduler: Scheduler, 
                     self_cond_prob: float = 0.5) -> Tensor:
        """
        Args:
            input_ids: [B, L] token indices
            mask: [B, L] binary mask (1 = noisy/generate, 0 = clean/condition)
            scheduler: noise scheduler
            self_cond_prob: probability of using self-conditioning during training
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Get clean embeddings
        embeddings = self.token_embeddings(input_ids)  # [B, L, D]
        
        # Sample timesteps
        t = scheduler.sample((batch_size,)).to(device)  # [B]
        
        # Expand mask to embedding dimension
        mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
        
        # Add noise only to masked positions
        noise = torch.randn_like(embeddings)
        noisy_embeddings = embeddings + noise * t.view(-1, 1, 1) * mask_expanded
        
        # Scale noisy embeddings (from paper section 6.1)
        scaling = 1.0 / torch.sqrt(t ** 2 + 1)
        noisy_embeddings = noisy_embeddings * scaling.view(-1, 1, 1)
        
        # Clean embeddings (conditioning)
        clean_embeddings = embeddings * (1 - mask_expanded)
        
        # Self-conditioning: with probability, use previous prediction
        prev_predictions = torch.zeros_like(embeddings)
        if self_cond_prob > 0 and torch.rand(1).item() < self_cond_prob:
            with torch.no_grad():
                # Get prediction without self-conditioning
                logits_prev = self.model(noisy_embeddings, clean_embeddings, 
                                        prev_predictions, mask, t)
                # Interpolate to get predicted embeddings
                prev_predictions = self.token_embeddings.interpolate_embeddings(logits_prev)
                prev_predictions = prev_predictions * mask_expanded  # Only for noisy positions
        
        # Forward pass with self-conditioning
        logits = self.model(noisy_embeddings, clean_embeddings, 
                          prev_predictions, mask, t)
        
        return logits
    
    @torch.no_grad()
    def denoise(self, noised_embeddings: Tensor, clean_embeddings: Tensor, 
                mask: Tensor, steps: int, scheduler: Scheduler) -> Tensor:
        """
        ODE-based sampling using Euler method.
        
        Args:
            noised_embeddings: [B, L, D] initial noisy embeddings
            clean_embeddings: [B, L, D] clean conditioning embeddings
            mask: [B, L, 1] binary mask
            steps: number of denoising steps
            scheduler: noise scheduler
        """
        device = noised_embeddings.device
        batch_size = noised_embeddings.size(0)
        
        # Create timesteps from tmax to tmin
        timesteps = scheduler.make_timesteps(steps + 1, tmax=scheduler.tmax)
        
        x_t = noised_embeddings.clone()
        prev_pred = torch.zeros_like(x_t)  # For self-conditioning
        
        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            
            # Current timestep tensor
            t_batch = torch.full((batch_size,), t_curr.item(), device=device)
            
            # Get model prediction with self-conditioning
            logits = self.model(x_t, clean_embeddings, prev_pred, mask, t_batch)
            
            # Interpolate to get predicted x_0
            x_0_pred = self.token_embeddings.interpolate_embeddings(logits)
            
            # Update self-conditioning for next step
            prev_pred = x_0_pred * mask
            
            # Compute score: s(x_t, t) = (x_0_pred - x_t) / t^2 (Equation 7 & 8)
            score = (x_0_pred - x_t) / (t_curr ** 2)
            
            # ODE derivative: dx/dt = t * score (Equation 5)
            dx_dt = t_curr * score
            
            # Euler step: x_{t+dt} = x_t + dx/dt * dt
            dt = t_next - t_curr
            x_t = x_t + dx_dt * dt
            
            # Keep clean embeddings unchanged
            x_t = x_t * mask + clean_embeddings * (1 - mask)
        
        # Final prediction at t_min
        t_final = torch.full((batch_size,), timesteps[-1].item(), device=device)
        logits = self.model(x_t, clean_embeddings, prev_pred, mask, t_final)
        
        return logits

    def generate(self, batch_size: int, steps: int, scheduler: Scheduler, 
                input_ids: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        """
        Generate tokens given optional conditioning.
        
        Args:
            batch_size: number of sequences to generate
            steps: number of denoising steps
            scheduler: noise scheduler
            input_ids: [B, L] optional conditioning tokens
            mask: [B, L] optional mask (1 = generate, 0 = condition)
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embed_dim = self.token_embeddings.token_embeddings.embedding_dim
        
        if mask is None:
            # Unconditional generation: all positions are noisy
            mask = torch.ones((batch_size, self.seq_len), device=device)
            clean_embeddings = torch.zeros(batch_size, self.seq_len, embed_dim, device=device)
            # Initialize with noise at t_max
            noised_embeddings = torch.randn(batch_size, self.seq_len, embed_dim, 
                                          device=device) * scheduler.tmax
        else:
            # Conditional generation
            if input_ids is None:
                raise ValueError("input_ids must be provided if mask is provided")
            
            # Get embeddings
            embeddings = self.token_embeddings(input_ids.to(device))
            
            # Expand mask if needed
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # [B, L, 1]
            
            # Clean embeddings are conditioning tokens
            clean_embeddings = embeddings * (1 - mask)
            
            # Initialize noisy positions with noise at t_max
            noise = torch.randn_like(embeddings) * scheduler.tmax
            noised_embeddings = embeddings * (1 - mask) + noise * mask
        
        # Run denoising process
        logits = self.denoise(noised_embeddings, clean_embeddings, mask, steps, scheduler)
        
        # Convert to token IDs
        probs = F.softmax(logits, dim=-1)
        generated_ids = torch.argmax(probs, dim=-1)
        
        # Combine generated and conditioning tokens
        if input_ids is not None:
            mask_2d = mask.squeeze(-1) if mask.dim() == 3 else mask
            generated_ids = generated_ids * mask_2d + input_ids * (1 - mask_2d)
        
        return generated_ids.to(torch.long)