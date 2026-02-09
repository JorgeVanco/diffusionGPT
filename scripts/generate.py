import sys
import os
import torch
from dataclasses import dataclass, field
from transformers import HfArgumentParser, AutoModelForMaskedLM, AutoTokenizer

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import TextDiffusionPipeline
from src.utils import animate_diffusion, visualize_stream
from src.logger import setup_logging

logger = setup_logging()

@dataclass
class GenerationArguments:
    model_path: str = field(
        metadata={"help": "Path to the model checkpoint or Hugging Face hub ID."}
    )
    prompt: str = field(
        default="Once upon a time,",
        metadata={"help": "Input text to start generation."}
    )
    mode: str = field(
        default="standard",
        metadata={"help": "Generation mode: 'standard' (fixed length) or 'semi_autoregressive' (long form)."}
    )
    num_steps: int = field(
        default=50,
        metadata={"help": "Number of diffusion steps."}
    )
    max_length: int = field(
        default=128,
        metadata={"help": "Total length of the generated sequence."}
    )
    block_size: int = field(
        default=64,
        metadata={"help": "Block size for semi-autoregressive generation."}
    )
    visualization: str = field(
        default="stream",
        metadata={"help": "Type of visualization: 'stream' (live updates) or 'animate' (replay after finish)."}
    )
    use_chat_template: bool = field(
        default=False,
        metadata={"help": "If True, wraps the prompt in the model's chat template as a user message."}
    )

def main():
    parser = HfArgumentParser((GenerationArguments,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # Allow passing a config file just like training
        args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    logger.info(f"🤖 Loading model from {args.model_path}...")
    
    try:
        model = AutoModelForMaskedLM.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except OSError:
        logger.error(f"Could not load model from {args.model_path}. Please check the path.")
        return

    pipe = TextDiffusionPipeline(model=model, tokenizer=tokenizer)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        pipe.model.to("cuda")
        logger.info("✅ Model moved to CUDA")

    # --- Chat Template Logic ---
    final_prompt = args.prompt
    if args.use_chat_template:
        logger.info("💬 Applying Chat Template...")
        messages = [{"role": "user", "content": args.prompt}]
        try:
            final_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            logger.error(f"Failed to apply chat template: {e}")
            logger.warning("Falling back to raw prompt.")
    
    logger.info(f"📝 Input Text: '{final_prompt}'")
    logger.info(f"⚙️ Mode: {args.mode}")

    # --- Mode: Semi-Autoregressive (Long Form) ---
    if args.mode == "semi_autoregressive":
        if args.visualization == "stream":
            logger.info("🔴 Starting Streaming Generation...")
            generator = pipe.stream_semi_autoregressive_generate(
                input_text=final_prompt,
                block_size=args.block_size,
                max_length=args.max_length,
                num_steps=args.num_steps
            )
            visualize_stream(generator)
        else:
            logger.info("⏳ Generating (please wait)...")
            output = pipe.semi_autoregressive_generate(
                input_text=final_prompt,
                block_size=args.block_size,
                max_length=args.max_length,
                num_steps=args.num_steps
            )
            print("\nFinal Output:\n" + "="*40)
            print(output["decoded_texts"][0])
            print("="*40)

    # --- Mode: Standard (Fixed Length) ---
    else: 
        if args.visualization == "stream":
            logger.info("🔴 Starting Streaming Generation...")
            generator = pipe.stream_generation(
                input_text=final_prompt,
                num_steps=args.num_steps,
                max_length=args.max_length
            )
            visualize_stream(generator)
        elif args.visualization == "animate":
            logger.info("⏳ Generating for Animation...")
            output = pipe(
                final_prompt,
                num_steps=args.num_steps,
                max_length=args.max_length
            )
            # Replay the generation history
            animate_diffusion(output, tokenizer)
        else:
            output = pipe(
                final_prompt,
                num_steps=args.num_steps,
                max_length=args.max_length
            )
            print("\nFinal Output:\n" + "="*40)
            print(output["decoded_texts"][0])
            print("="*40)

if __name__ == "__main__":
    main()