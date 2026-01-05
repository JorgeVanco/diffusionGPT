from dataclasses import dataclass, field
from typing import Optional, Dict
from transformers import TrainingArguments

@dataclass
class DiffusionTrainingArguments(TrainingArguments):
    num_diffusion_steps: int = field(
        default=10,
        metadata={"help": "Number of denoising steps for generation/evaluation."}
    )
    corruption_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of replacing a visible token with a random word during training."}
    )
    auto_naming: bool = field(
        default=False,
        metadata={"help": "If true, the output_dir will be automatically named based on hyperparameters."}
    )
    edit_stage_start: float = field(
        default=0.8,
        metadata={"help": "Fraction of training after which to start the edit-based training stage."}
    )
    anneal_corruption: bool = field(
        default=True,
        metadata={"help": "Whether to anneal the corruption probability during the edit stage."}
    )
    target_param_data_ratio: Optional[int] = field(
        default=None,
        metadata={"help": "If set, adjusts the number of training samples to achieve the target parameter-to-data ratio."}
    )

    
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="answerdotai/ModernBERT-base", 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: str = field(
        default="gpt2",
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    hidden_size: int = field(default=256, metadata={"help": "Hidden dimension size"})
    num_hidden_layers: int = field(default=4, metadata={"help": "Number of layers"})
    num_attention_heads: int = field(default=4, metadata={"help": "Number of attention heads"})
    special_tokens: Optional[Dict[str, str]] = field(
        default_factory=lambda: {"pad_token": "<pad>", "mask_token": "<mask>", "eos_token": "<eos>"},
        metadata={"help": "Dictionary of special tokens to add/override in the tokenizer."}
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        default="roneneldan/TinyStories",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_subset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The subset name of the dataset to use (via the datasets library)."}
    )
    streaming: bool = field(
        default=False, 
        metadata={"help": "If True, streams data (good for FineWeb). If False, downloads and caches (good for TinyStories)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of evaluation examples."}
    )
    load_from_disk: Optional[str] = field(
        default=None,
        metadata={"help": "If set, loads the dataset from the specified local disk path."}
    )