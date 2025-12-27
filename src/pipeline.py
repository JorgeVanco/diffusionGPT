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
            
        if input_text is None or input_text == "":
            input_ids = torch.full((1, max_length), self.tokenizer.mask_token_id, dtype=torch.long) # type: ignore
            return BatchEncoding({
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids)
            })

        return self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
    
    @torch.no_grad()
    def _forward(self, model_inputs, num_steps=10) -> dict[str, Any]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer was not passed to the pipeline!")
        
        current_state = model_inputs["input_ids"]
        all_states = [current_state.clone()]
        
        # Determine which tokens can be re-masked (i.e., mask and pad tokens)
        remasking_mask = (current_state == self.tokenizer.mask_token_id) | (current_state == self.tokenizer.pad_token_id)
        
        for step in range(num_steps):
            t_current = 1 - step / num_steps
            t_next = 1 - (step + 1) / num_steps
            
            # Predict full text with model
            output = self.model(input_ids=current_state)
            logits = output.logits
            
            # Set logit that corresponds to the mask token to -inf
            logits[:, :, self.tokenizer.mask_token_id] = torch.finfo(logits.dtype).min
            
            # Ancestral sampling logic
            probs = torch.softmax(logits, dim=-1)
            sampled_ids = torch.distributions.Categorical(probs).sample()
            
            # Calculate Unmasking Probability (Equation 7 https://arxiv.org/pdf/2406.07524)
            # P(unmask | masked) = (alpha_s - alpha_t) / (1 - alpha_t)
            # mapping: alpha_t = (1 - t_current), alpha_s = (1 - t_next)
            # resulting simplified formula: (t_current - t_next) / t_current
            if step < num_steps - 1:
                unmasking_prob = (t_current - t_next) / t_current
            else:
                unmasking_prob = 1.0 # Force unmask at the end
            
            # Unmask the tokens if unmasking_mask is True
            unmasking_mask = torch.rand(current_state.shape, device=current_state.device) < unmasking_prob
            
            remasking_mask &= (current_state == self.tokenizer.mask_token_id) | (current_state == self.tokenizer.pad_token_id)
            
            update_mask = unmasking_mask & remasking_mask
            
            # Update current state
            current_state[update_mask] = sampled_ids[update_mask]
            
            all_states.append(current_state.clone())
        
        return {"final_state": current_state, "history": all_states}
        
    def postprocess(self, model_outputs) -> list[str] | Any:
        if self.tokenizer is None:
            raise ValueError("Tokenizer was not passed to the pipeline!")
        
        # Convert final tensor to image/text
        final_ids = model_outputs["final_state"]
        return {
            "decoded_texts": self.tokenizer.batch_decode(final_ids, skip_special_tokens=False),
            "history": model_outputs["history"],
            "final_ids": final_ids
        }