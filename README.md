<div align="center">
<img src="./public/DiffusionGPT_logo.png">

### A Discrete Diffusion Language Model for Conversational AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/JorgeVanco/diffusionGPT)

[**Quick Start**](#quick-start) • [**Documentation**](#features) • [**Research**](#research-background) • [**Training**](#training)

---

</div>

https://github.com/user-attachments/assets/84ce6d3f-603f-4940-8034-3b961c9778e7

## Overview

**diffusionGPT** is a novel language model that generates text through iterative refinement rather than sequential prediction. Unlike traditional autoregressive models (GPT, Llama) that generate text left-to-right one token at a time, diffusionGPT uses a **discrete diffusion process** to simultaneously denoise and refine entire sequences.

This approach enables:

- **Parallel token generation** across the sequence
- **Creative text editing** and refinement capabilities
- **Flexible generation modes** (standard and semi-autoregressive)
- **Interpretable generation** - watch the model "think" in real-time

### See It In Action

The following video shows the semi-autoregressive generation of a small model trained on TinyStories.

https://github.com/user-attachments/assets/51cd28be-8111-4c1d-9a89-faeaa22e97bf

---

## Features

### Parallel Decoding

Unlike autoregressive models that must generate tokens sequentially, diffusionGPT generates and refines multiple tokens simultaneously, allowing for more flexible and potentially faster generation strategies.

### Seed Diffusion Editing

Implements editing logic from [Seed Diffusion](https://arxiv.org/abs/2508.02193), enabling the model to refine and improve existing text while maintaining semantic coherence.

### Flexible Generation Modes

1. **Standard Diffusion**: Full-sequence generation with configurable denoising steps
2. **Semi-Autoregressive**: Block-wise generation for long-form content that scales beyond the model's context window
3. **Streaming**: Real-time visualization of the denoising process

### Custom Pipeline Architecture

Built-in `TextDiffusionPipeline` with the following features:

- Ancestral sampling with configurable noise schedules
- Confidence-based token unmasking
- Stop token detection
- Block-wise generation support

---

## Quick Start

The model can be found at FuggingFace [JorgeVanco/diffusionGPT](https://huggingface.co/JorgeVanco/diffusionGPT).

### Installation

```bash
# Clone the repository
git clone https://github.com/JorgeVanco/diffusionGPT.git
cd diffusionGPT

# Install dependencies with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Basic Usage

```python
from transformers import pipeline

# Load the model
pipe = pipeline(
    "text-diffusion",
    model="JorgeVanco/diffusionGPT",
    trust_remote_code=True
)

# Prepare your prompt
messages = [
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]
prompt = pipe.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Generate response
result = pipe(prompt, num_steps=50)
print(result["decoded_texts"][0])
```

### Streaming Generation

Watch the model refine text in real-time:

```python
# Stream the denoising process
for step_text in pipe.stream_generation(prompt, num_steps=32, allow_edits=True):
    print(f"\033[H\033[J{step_text}")  # Clear and update terminal
```

### Semi-Autoregressive Generation

For generating longer sequences beyond the model's context window:

```python
# Generate long-form content block-by-block
for step_text in pipe.stream_semi_autoregressive_generate(
    input_text=prompt,
    block_size=128,
    max_length=2048,
    num_steps=32
):
    print(step_text)
```

### Interactive Chat Interface

Launch the Chainlit-based chat interface:

```bash
uv run chainlit run app.py
```

This provides a user-friendly web interface with:

- Real-time generation visualization
- Adjustable diffusion parameters
- Support for both generation modes
- Custom system prompts

---

## Research Background

diffusionGPT builds upon recent advances in discrete diffusion for natural language:

### Masked Diffusion Language Models (MDLM)

The core training methodology is based on [MDLM](https://s-sahoo.com/mdlm/), which formulates text generation as an iterative denoising process. At each training step:

1. **Forward Process**: Randomly mask a portion of input tokens
2. **Reverse Process**: Train the model to predict the original tokens from the corrupted input
3. **Time Weighting**: Apply time-dependent loss weighting for stable training

#### The Diffusion Algorithm

The model learns to reverse a corruption process:

1. **Training Time**:
    - Sample timestep `t ~ Uniform(0, 1)`
    - Mask proportion `t` of tokens → `input_ids_noisy`
    - Train model to predict original tokens from `input_ids_noisy`

2. **Inference Time**:
    - Start with fully masked sequence
    - Iteratively unmask tokens based on model predictions
    - Apply Seed Diffusion editing to refine visible tokens
    - Continue until all tokens are revealed

### Seed Diffusion Editing

We implement the robust training curriculum from [Seed Diffusion](https://arxiv.org/abs/2508.02193):

- **Stage 1 (0-80% of training)**: Standard MDLM objective
- **Stage 2 (80-100% of training)**: Introduce controlled corruption to visible tokens, teaching the model to refine and edit existing text

This two-stage approach significantly improves generation quality and enables the model to perform coherent text editing.

---

## The Creation Process

Building diffusionGPT involved translating theoretical concepts from discrete diffusion papers into a practical, Hugging Face-compatible ecosystem. The development process followed four distinct stages:

### 1. Architecture Selection

Instead of using a standard decoder-only model (like Llama or GPT), this project utilizes **ModernBERT** (Answer.AI) as the backbone.

- **Reasoning**: Diffusion models require bidirectional attention to "see" the entire sequence context while refining masked tokens. Encoder-only architectures are naturally suited for this.
- **Implementation**: The configuration is adapted in [`src/argument_classes.py`](src/argument_classes.py) to support `ModernBERT-base` (~600M params) with a custom vocabulary size.
- **Specs**:
    - **Context Length**: 2048 tokens
    - **Vocabulary**: GPT-2 tokenizer
    - **Special Tokens**: `<|im_start|>`, `<|im_end|>`, `<mask>`, `<|delete|>` (for insertion corruption)

### 2. Custom Training Logic

Standard causal language modeling (CLM) trainers cannot handle diffusion. I implemented a custom training loop centered around:

- **The Collator**: A [`DiscreteDiffusionCollator`](src/trainer.py) was built to handle dynamic masking logic ($t \sim U(0,1)$) and label creation on the fly. This data collator masks the inputs, applies the edit corruption if necessary and also applies any additional masks for _sft_.
- **The Trainer**: The [`DiffusionTrainer`](src/trainer.py) overrides the loss computation to apply time-dependent weighting, ensuring the model focuses on getting right the last stages of denoising.

> **Note:** I wanted to use the Huggingface ecosystem instead of doing everything from scratch to learn how to use the existing libraries.

### 3. Implementing Seed Diffusion (Curriculum Learning)

To support editing and higher consistency, I implemented the "Seed Diffusion" two-stage training strategy via custom callbacks:

- **Stage 1 (MDLM)**: Pure masking and reconstruction.
- **Stage 2 (Interpolation)**: Facilitated by the [`SeedDiffusionCurriculumCallback`](src/trainer_callbacks.py), which continuously monitors training steps. Once the `edit_stage_start` threshold (default 80%) is reached, it introduces "gold" token corruption, teaching the model to refine existing incorrect tokens rather than just filling blanks. It also introduces the `<|delete|>` token, enabling the model to learn token removal (note: delete corruption is not yet implemented for _sft_).

> **Note:** In my experiments I found that having the edit stage active for the whole length of the training yielded better results. I believe this is due to making better use of the limited number of tokens, as this stage uses all the tokens for the loss calculation instead of just the handful of masked tokens.

### 4. The Inference Engine

For model inference [`TextDiffusionPipeline`](src/pipeline.py) is used. Since there is no standard "generate" method for diffusion models in Transformers, this custom pipeline handles:

- **Ancestral Sampling**: Iteratively sampling $x_{t-1}$ from $x_t$ using the model's logits.
- **Confidence-Based Unmasking**: Logic to selectively unmask only the most "confident" tokens at each step.
- **Semi-Autoregressive Blocks**: A wrapper method that allows generating text beyond the context window by treating the previous generation as a fixed context for the next block.

> **Note:** I found that using random unmasking works better than confidence-based unmasking, but I have to dive deeper into this and improve the confidence-based unmasking procedure.

### 5. Hyperparameter Optimization

To find the optimal balance between noise schedules and learning rates, the project includes an Optuna-based sweep script ([`sweep.py`](sweep.py)) capable of running distributed trials across multiple GPUs.

### 6. Model Training

First, I trained a small model on the `TinyStories` dataset. It was not perfect, but could generate reasonable stories. Then, I scaled up to the `FineWeb` dataset. However, it doesn't generate much coherent text. I returned to debugging and testing on the `TinyStories` dataset but everything seemed okay, probably I just have to find better hyperparameters. In hindsight, I should have performed the sweep instead of training it fully just to see if it works. I will do some sweeping but first I want to implement some benchmarks to not rely only on the loss.

### 7. Supervised Fine-Tuning

I did a test on the `smol-smoltalk` dataset to see if the model would generate coherent responses and it seem to somewhat work, so I continued to work on _sft_ hoping that it would at least work on some conversations. As the model does not have a causal mask, I split the conversations so that the model only sees the previous part and what it has to answer. First, I trained on the full conversation, not just the assistant response. The model learned to generate coherent answers, it also worked with the semi-autoregressive inference mode. However, it got confused many times and asked as a user instead of answering as an assistant. I then used the assistant masks extracted from the chat-template to train only on the assistant answers. That seemed to work better, so I added the `everyday-conversations-llama3.1-2k` and `Nemotron-Instruction-Following-Chat-v1` datasets. When finished training, the model could answer questions reliably. However, it always ended with the `<|im_end|>` token, regardless of the length of its input. With the padding and masks, the model always "sees" the `<|im_end|>` token at the end, so it always generates it at the end, no matter the length of the text. The upside is that you can select the desired answer length, the downside is that you no longer have the semi-autoregressive capabilities. This is something I have to look into.

## Training

### Training from Scratch

The repository includes complete training scripts with support for:

- Multi-GPU distributed training
- Streaming datasets (FineWeb, etc.)
- WandB integration
- Custom learning rate schedules
- Automatic checkpoint management

#### Basic Training Command

```bash
# Train with default configuration
uv run accelerate launch train.py configs/nanochat_llm.yaml

# Train with custom parameters
uv run train.py \
    --num_hidden_layers 12 \
    --hidden_size 768 \
    --num_diffusion_steps 100 \
    --max_seq_length 2048 \
    --learning_rate 4e-4 \
    --target_param_data_ratio 40
```

#### Supervised Fine-Tuning

Fine-tune on conversational data:

```bash
# Preprocess chat datasets
uv run preprocess_chat_dataset.py configs/sft.yaml

# Run SFT
uv run accelerate launch sft.py configs/sft.yaml
```

### Hyperparameter Optimization

We include Optuna-based hyperparameter sweeps with parallel GPU support:

```bash
# Launch distributed sweep across 8 GPUs
bash scripts/sweep.sh

# Monitor optimization progress
optuna-dashboard sqlite:///db.sqlite3
```

### Training Configuration

Key hyperparameters (see `configs/` for full details):

| Parameter              | Default | Description                              |
| ---------------------- | ------- | ---------------------------------------- |
| `num_diffusion_steps`  | 100     | Steps during training diffusion          |
| `corruption_prob`      | 0.1     | Token corruption probability             |
| `edit_stage_start`     | 0.8     | When to begin Seed Diffusion editing     |
| `anneal_corruption`    | True    | Gradually increase corruption in stage 2 |
| `insertion_corruption` | True    | Use insertion-based corruption           |
| `time_loss_weighting`  | True    | Apply MDLM time weighting                |

---

## Model Performance

### Generation Quality

The model has been trained on a mixture of:

- **Pre-training**: FineWeb-100BT (not trained with the whole dataset)
- **Fine-tuning**: SmolTalk, Everyday Conversations, Nemotron datasets

---

## Generation Parameters

Fine-tune generation behavior with these parameters:

```python
result = pipe(
    prompt,
    num_steps=50,           # More steps = higher quality, slower
    allow_edits=True,       # Enable Seed Diffusion editing
    use_confidence=False,   # Use confidence-based vs random unmasking
    stop_token="<|im_end|>" # Early stopping on special token
)
```

| Parameter        | Type | Default | Description                                   |
| ---------------- | ---- | ------- | --------------------------------------------- |
| `num_steps`      | int  | 50      | Number of denoising iterations                |
| `allow_edits`    | bool | True    | Enable iterative refinement of visible tokens |
| `use_confidence` | bool | False   | Unmask highest-confidence tokens first        |
| `block_size`     | int  | 128     | Tokens per block (semi-autoregressive mode)   |
| `max_length`     | int  | 2048    | Maximum sequence length to generate           |

---

## Project Structure

```
diffusionGPT/
├── app.py                      # Chainlit chat interface
├── train.py                    # Pre-training script
├── sft.py                      # Supervised fine-tuning script
├── configs/
│   ├── config.yaml            # Base training config
│   ├── nanochat_llm.yaml      # LLM-scale configuration
│   └── sft.yaml               # Fine-tuning config
├── src/
│   ├── pipeline.py            # TextDiffusionPipeline implementation
│   ├── trainer.py             # Custom DiffusionTrainer
│   ├── trainer_callbacks.py  # Training callbacks
│   ├── data_utils.py          # Dataset loading utilities
│   └── utils.py               # Helper functions
├── tasks/                     # Dataset task definitions
└── scripts/
    ├── sweep.sh              # Multi-GPU hyperparameter search
    ├── prepare_model_upload.py
    └── upload_model.py
```

---

## Limitations

- **Factuality**: Like all LLMs, can produce hallucinations. Not optimized for factual retrieval.
- **Long-range Coherence**: Most effective for short-to-medium conversations. Long-form coherence is an active development area.
- **Speed vs Quality Trade-off**: Fewer diffusion steps = faster but lower quality. Tuning required for your use case.
- **Training Data**: Primarily trained on English conversational data.

---

## Roadmap

- [ ] **Fix Semi-Autoregressive SFT**: Modify training data/masking so the model isn't "forced" to end every sequence with `<|im_end|>`, enabling true block-wise generation for long texts.
- [ ] **SFT Improvements**: Implement "deletion corruption" logic during the fine-tuning stage.
- [ ] **Confidence-based sampling**: Improve the current basic implementation of confidence-based sampling.
- [ ] **Evaluation Suite**: Implement proper benchmarks (beyond loss) to evaluate generation quality quantitatively.
- [ ] **Hyperparameter Tuning**: Run comprehensive sweeps on `FineWeb` to find the sweet spot for learning rates and noise schedules.
- [ ] **Future Features**:
    - [ ] Tool use / Function calling
    - [ ] "Thinking" capability (Chain of Thought)
    - [ ] Quantization for edge deployment

---

## Acknowledgments

This implementation builds upon excellent prior work:

- **MDLM**: [Simple and Effective Masked Diffusion Language Models](https://s-sahoo.com/mdlm/)
- **Seed Diffusion**: [Seed Diffusion: Continuous Training of Discrete Diffusion Language Models](https://seed.bytedance.com/en/seed_diffusion)
- **NanoChat**: [Andrej Karpathy's nanochat repo](https://github.com/karpathy/nanochat)
- **ModernBERT**: Efficient transformer architecture by Answer.AI
- **Hugging Face Transformers**: Foundational infrastructure

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

- **GitHub**: [@JorgeVanco](https://github.com/JorgeVanco)
- **Model**: [JorgeVanco/diffusionGPT](https://huggingface.co/JorgeVanco/diffusionGPT)

---
