from transformers import BatchEncoding, Pipeline
import torch
from typing import Any

from src.utils import mask_input_ids_

class TextDiffusionPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        # Allow user to control the number of steps (e.g., diffusion steps)
        # default to 10 steps
        forward_kwargs = {"num_steps": kwargs.get("num_steps", 10)}
        return {}, forward_kwargs, {}
    
    def preprocess(self, input_text, max_length=None) -> BatchEncoding | Any:
        if self.tokenizer is None:
            raise ValueError("Tokenizer was not passed to the pipeline!")
        # Standard tokenization
        if max_length is None:
            # Safely access config if it exists, default to 512
            max_length = getattr(self.model.config, "n_positions", 512)

        return self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
    
    def _forward(self, model_inputs, num_steps=10) -> dict[str, Any]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer was not passed to the pipeline!")
        
        current_state = model_inputs["input_ids"]
        all_states = [current_state.clone()]
        for step in range(num_steps):
            with torch.no_grad():
                # Predict full text with model
                output = self.model(input_ids=current_state)
                logits = output.logits
                
                pred_ids = torch.argmax(logits, dim=-1)
                current_state = pred_ids
            
            # Re-mask for next step, except on last step
            if step < num_steps - 1:
                # Mask corresponding portion of text
                t = 1 - (step + 1) / num_steps
                mask_input_ids_(
                    current_state,
                    mask_token_id=self.tokenizer.mask_token_id,
                    mask_prob=torch.full((current_state.size(0),), t, device=current_state.device),
                    generator=None
                )
            all_states.append(current_state.clone())
        
        return {"final_state": current_state, "history": all_states}
        
    def postprocess(self, model_outputs) -> list[str] | Any:
        if self.tokenizer is None:
            raise ValueError("Tokenizer was not passed to the pipeline!")
        
        # Convert final tensor to image/text
        final_ids = model_outputs["final_state"]
        return self.tokenizer.batch_decode(final_ids, skip_special_tokens=False)