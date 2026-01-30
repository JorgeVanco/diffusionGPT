# Make sure to first run `uv run -m scripts.prepare_model_upload` to create the temp_upload folder
# uv run -m scripts.upload_model


from huggingface_hub import login, upload_folder

login()

# Upload the folder containing model, tokenizer, and pipeline.py
upload_folder(
    folder_path="./temp_upload",
    repo_id="JorgeVanco/diffusionGPT",
    repo_type="model"
)