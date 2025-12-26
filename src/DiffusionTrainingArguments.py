from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class DiffusionTrainingArguments(TrainingArguments):
    num_diffusion_steps: int = field(
        default=10,
        metadata={"help": "Number of denoising steps for generation/evaluation."}
    )