import chainlit as cl
from transformers import pipeline
# from src.pipeline import TextDiffusionPipeline
# Ensure your src folder is accessible or copy the class here
# from preprocess_chat_dataset import setup_chat_format # If you need this

# --- 1. GLOBAL MODEL LOADING ---
# We load the model once at the start of the server
@cl.cache
def load_model():
    # checkpoint_path = "./output/diffusion-baseline-sft/"
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # print(f"Loading model from {checkpoint_path} on {device}...")
    # model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    # # tokenizer = setup_chat_format(tokenizer)
    # # model.resize_token_embeddings(len(tokenizer))
    # model.to(device)

    # # Initialize your custom pipeline
    # pipe = TextDiffusionPipeline(model=model, tokenizer=tokenizer, device=device if isinstance(device, int) else 0)
    pipe = pipeline(
        "text-diffusion",
        model="JorgeVanco/diffusionGPT",
        trust_remote_code=True
    )
    return pipe

diffusion_pipeline = load_model()

# --- 2. CHAT SETTINGS (Sidebar) ---
@cl.on_chat_start
async def start():
    # Create inputs for the user to control diffusion parameters
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Slider(
                id="num_steps",
                label="Diffusion Steps",
                initial=100,
                min=10,
                max=500,
                description="More steps = higher quality, slower generation."
            ),
            cl.input_widget.Slider(
                id="block_size",
                label="Block Size",
                initial=128,
                min=32,
                max=1024,
                description="Larger blocks = more generated at once, but slower generation."
            ),
            cl.input_widget.Switch(
                id="allow_edits",
                label="Allow Edits (Seed Diffusion)",
                initial=True,
            ),
             cl.input_widget.Select(
                id="mode",
                label="Generation Mode",
                values=["Standard Diffusion", "Semi-Autoregressive"],
                initial_index=1
            ),
            cl.input_widget.TextInput(
                id="system_prompt",
                label="System Prompt",
                initial="You are a helpful and charming assistant.",
                multiline=True,
                description="The system prompt sets the behavior of the assistant."
            )
        ]
    ).send()

    cl.chat_context.add(
        cl.Message(
            content=settings["system_prompt"],
            type="system_message"
        )
    )

# --- 3. MESSAGE HANDLER ---
@cl.on_message
async def main(message: cl.Message):
    # 1. Get current history and settings
    settings = cl.user_session.get("chat_settings")

    # Provide defaults if settings haven't been touched yet
    num_steps = int(settings["num_steps"]) if settings else 100
    block_size = int(settings["block_size"]) if settings else 128
    allow_edits = bool(settings["allow_edits"]) if settings else True
    mode = settings["mode"] if settings else "Standard Diffusion"

    messages = cl.chat_context.to_openai()

    # 3. Apply Chat Template to the ENTIRE history
    full_prompt = diffusion_pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 4. Prepare the empty message container
    # We send an empty message first so we have an ID to update
    msg = cl.Message(content="")
    await msg.send()

    # 5. Stream the Diffusion Process
    try:
        if mode == "Standard Diffusion":
            # print(  f"Starting Standard Diffusion Generation for prompt: {full_prompt} "  )
            generator = diffusion_pipeline.stream_generation(
                full_prompt,
                num_steps=num_steps,
                allow_edits=allow_edits,
                stop_token="<|im_end|>"
            )
        else:
            # Hardcoded block params for the demo, or add them to settings
            generator = diffusion_pipeline.stream_semi_autoregressive_generate(
                full_prompt,
                num_steps=num_steps,
                block_size=block_size,
                max_length=20000,
                allow_edits=allow_edits,
                stop_token="<|im_end|>"
            )

        # LOOP: Update the same message repeatedly
        for i, step_text in enumerate(generator):
            # Optional: Visual flair - make masks look like blocks
            step_text = step_text[len(full_prompt):]  # Remove prompt part
            display_text = step_text.replace(
                "<mask>",
                '<span style="color: #F02E65; font-weight: bold;">▓</span>'
            )

            # Update the UI
            msg.content = f'{display_text}'
            await msg.update()


            # Optional: Add a subtle delay if it's too fast to see
            # await cl.sleep(0.02)

    except Exception as e:
        msg.content = f"**Error**: {str(e)}"
        await msg.update()


# --- 4. UPDATE SETTINGS CALLBACK ---
@cl.on_settings_update
async def setup_agent(settings):
    # Store settings in session so the message handler can access them
    cl.user_session.set("chat_settings", settings)