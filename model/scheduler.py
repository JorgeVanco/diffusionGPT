import torch   


class Scheduler:
    """
    Noise scheduler for CDCD.
    
    Following the paper (Section 6.1):
    - For embed_dim=256: t_min=10, t_max=3000
    - Timesteps are sampled uniformly during training
    - During sampling, timesteps are linearly spaced from t_max to t_min
    """
    def __init__(self, tmin: float = 10.0, tmax: float = 3000.0) -> None:
        self.tmin = tmin
        self.tmax = tmax

    def sample(self, shape: torch.Size) -> torch.Tensor:
        """Sample timesteps uniformly for training."""
        return torch.rand(shape) * (self.tmax - self.tmin) + self.tmin
    
    def make_timesteps(self, steps: int, tmax: float | None = None) -> torch.Tensor:
        """
        Create linearly spaced timesteps for sampling.
        
        Args:
            steps: number of timesteps
            tmax: optional override for max timestep
            
        Returns:
            Tensor of shape [steps] with timesteps from tmax to tmin
        """
        if tmax is None:
            tmax = self.tmax
        return torch.linspace(tmax, self.tmin, steps)
    
    def log_uniform_sample(self, shape: torch.Size) -> torch.Tensor:
        """
        Alternative: sample timesteps log-uniformly.
        This can be more efficient as it focuses on important noise levels.
        """
        log_tmin = torch.log(torch.tensor(self.tmin))
        log_tmax = torch.log(torch.tensor(self.tmax))
        log_t = torch.rand(shape) * (log_tmax - log_tmin) + log_tmin
        return torch.exp(log_t)