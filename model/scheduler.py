import torch   
    
class Scheduler:
    def __init__(self, tmin: float, tmax: float) -> None:
        self.tmin = tmin
        self.tmax = tmax

    def sample(self, shape: torch.Size) -> torch.Tensor:
        return torch.rand(shape) * (self.tmax - self.tmin) + self.tmin
    
    def make_timesteps(self, steps: int, tmax: float | None = None) -> torch.Tensor:
        if tmax is None:
            tmax = self.tmax
        return torch.linspace(tmax, self.tmin, steps)