from transformers import Pipeline
import torch
from typing import Any

from src.utils import mask_input_ids_

class TextDiffusionPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        # Allow user to control the number of steps (e.g., diffusion steps)
        # default to 10 steps
        forward_kwargs = {"num_steps": kwargs.get("num_steps", 10)}
        return {}, forward_kwargs, {}
    
    def preprocess(self, input_text):
        # Standard tokenization
        return self.tokenizer(input_text, return_tensors="pt")
    
    def _forward(self, model_inputs, num_steps=10) -> dict[str, Any]:
        current_state = model_inputs["input_ids"]
        all_states = [current_state]
        for step in range(num_steps):
            # Predict full text with model
            current_state = self.model(current_state)
            
            if step < num_steps - 1:
                # Mask corresponding portion of text
                t = 1 - (step + 1) / num_steps
                mask_input_ids_(
                    current_state,
                    mask_token_id=self.tokenizer.mask_token_id,
                    mask_prob=torch.full((current_state.size(0),), t, device=current_state.device),
                    generator=None
                )
            all_states.append(current_state)        
        
        return {"final_state": current_state, "history": all_states}
        
    def postprocess(self, model_outputs):
        # Convert final tensor to image/text
        return self.model.decode(model_outputs["final_state"])