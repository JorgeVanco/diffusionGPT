from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, GPT2Config

# train_dataset = load_dataset("roneneldan/TinyStories", split="train[:1]")
# # eval_dataset = load_dataset("roneneldan/TinyStories", split="validation")
# print(train_dataset[0])
# print(len(train_dataset))
# print(len(train_dataset[0]['text']))

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# tokenizer.mask_token = "<mask>"
# tokenizer.add_special_tokens({'mask_token': tokenizer.mask_token})
    
# print("Vocab size:", len(tokenizer))

# tokenizer("Hello world!", return_tensors="pt",padding="max_length",)

from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizer
from src.pipeline import TextDiffusionPipeline
from src.utils import animate_diffusion

# Instead of "bert-base-uncased", point to your folder
checkpoint_path = "./output/layers6_embd512_seq512_diff100_lr0.0004725735401380809_1231_1905/checkpoint-20000"

# Load exactly as you would a public model
model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Now use them normally
# pipe = TextDiffusionPipeline(model=model, tokenizer=tokenizer)
prompt = "Once upon a time in a land far, far away,<mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask><mask> and the big dragons"
prompt = "Once upon a time in a land far, far away,"
# import time
# t0 = time.time()
# output = pipe(prompt, num_steps=50, max_length=512)
# print(f"Generation took: {(time.time() - t0)*1000:.2f} milliseconds")
# animate_diffusion(output, tokenizer, interval=0.0, sample_idx=0)


# tokenzier = PretrainedTokenizer.from_pretrained("gpt2")
# tokenizer("Hello world!", return_tensors="pt", )

import time
import os
from IPython.display import clear_output

# Initialize your pipeline
pipe = TextDiffusionPipeline(model=model, tokenizer=tokenizer)
prompt = "Once upon a time,"

# print("Starting Real-Time Diffusion...\n")

# # Use the new streaming method
# generator = pipe.stream_generation(prompt, num_steps=50, max_length=128)

# for step, text in enumerate(generator):
#     # --- VISUALIZATION LOGIC ---
    
#     # Method A: For Jupyter Notebooks (flicker-free update)
#     # clear_output(wait=True)
#     # print(f"Step {step}:\n{text}")
    
#     # Method B: For Terminal (Matrix style)
#     os.system('cls' if os.name == 'nt' else 'clear') 
    
#     # Colorize MASKs for effect
#     colored_text = text.replace("<mask>", "\033[91m█\033[0m") # Red blocks for masks
    
#     print(f"Step {step:02d}")
#     print("-" * 40)
#     print(colored_text)
#     print("-" * 40)
    
#     # time.sleep(0.05) # Add small delay to see the animation

# print("\nDone!")

generator = pipe.stream_semi_autoregressive_generate(
    input_text=prompt, block_size = 256, max_length = 1024)
print("\nSemi-Autoregressive Generation Result:")
# print(out)

for step, text in enumerate(generator):
    # --- VISUALIZATION LOGIC ---
    
    # Method A: For Jupyter Notebooks (flicker-free update)
    # clear_output(wait=True)
    # print(f"Step {step}:\n{text}")
    
    # Method B: For Terminal (Matrix style)
    os.system('cls' if os.name == 'nt' else 'clear') 
    
    # Colorize MASKs for effect
    colored_text = text.replace("<mask>", "\033[91m█\033[0m") # Red blocks for masks
    
    print(f"Step {step:02d}")
    print("-" * 40)
    print(colored_text)
    print("-" * 40)
    
    # time.sleep(0.05) # Add small delay to see the animation
    
    
# out = pipe.semi_autoregressive_generate(
#     input_text="Once upon a time,", block_size = 64, max_length = 512)
# print("Final Output:\n", out['decoded_texts'][0])