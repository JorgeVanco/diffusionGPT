# Prepare the model and tokenizer for upload by adding custom pipeline info to the config
# uv run -m scripts.prepare_model_upload

from transformers import AutoModelForMaskedLM, AutoTokenizer

# Load your model and tokenizer
model_path = "./output/diffusion-baseline-sft/"
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Add the custom pipeline mapping to the config
# "text-diffusion" is your custom task name
# "TextDiffusionPipeline" is the class name in your pipeline.py
model.config.custom_pipelines = {
    "text-diffusion": {
        "impl": "pipeline.TextDiffusionPipeline",
        "pt": ("AutoModelForMaskedLM",),
        "tf": (),
    }
}

# Save them locally first to ensure the config is updated
model.save_pretrained("./temp_upload")
tokenizer.save_pretrained("./temp_upload")