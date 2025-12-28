from transformers import BatchEncoding, Pipeline
import torch
from typing import Any, Generator

from src.utils import mask_input_ids_

class TextDiffusionPipeline(Pipeline):
    def _sanitize_parameters(self, num_steps: int = 50, allow_edits: bool = True, **kwargs) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        # Allow user to control the number of steps (e.g., diffusion steps)
        # default to 10 steps
        forward_kwargs = {"num_steps": num_steps, "allow_edits": allow_edits}
        
        preprocess_kwargs = {}
        if "max_length" in kwargs:
            preprocess_kwargs["max_length"] = kwargs["max_length"]

        return preprocess_kwargs, forward_kwargs, {}
    
    def preprocess(self, input_text, max_length=None) -> BatchEncoding | Any:
        if self.tokenizer is None:
            raise ValueError("Tokenizer was not passed to the pipeline!")
        # Standard tokenization
        if max_length is None:
            # Safely access config if it exists, default to 512
            max_length = getattr(self.model.config, "seq_length", 512)
            
        if input_text is None:
            input_text = ""
            
        tokenized_text = self.tokenizer.encode(
            input_text
            )
        if len(tokenized_text) < max_length:
            input_ids = torch.full((1, max_length), self.tokenizer.mask_token_id, dtype=torch.long) # type: ignore
            input_ids[0, :len(tokenized_text)] = torch.tensor(tokenized_text, dtype=torch.long)

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
    def diffusion_generator(self, input_ids, num_steps, allow_edits=True):
        if self.tokenizer is None:
            raise ValueError("Tokenizer was not passed to the pipeline!")
        
        current_state = input_ids.clone()
        yield current_state.clone() # Yield Step 0
        
        # Determine which tokens can be re-masked (i.e., mask and pad tokens)
        initial_mask = (current_state == self.tokenizer.mask_token_id) | \
                       (current_state == self.tokenizer.pad_token_id)
        
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
            unmasking_mask = torch.rand_like(current_state, dtype=torch.float) < unmasking_prob
            
            remasking_mask = (current_state == self.tokenizer.mask_token_id) | \
                             (current_state == self.tokenizer.pad_token_id)
            
            update_mask = unmasking_mask & remasking_mask & initial_mask
            
            if allow_edits: # Apply Seed Diffusion Editing Logic (Section 3.1 in https://arxiv.org/pdf/2508.02193)
                alpha_t = 0.1 * (1 - step / num_steps)  # alpha_t decreases from 0.1 to 0 (Seed Diffusion)
                
                edit_mask = torch.rand_like(current_state, dtype=torch.float) < alpha_t
                
                is_visible = (current_state != self.tokenizer.mask_token_id) & \
                             (current_state != self.tokenizer.pad_token_id) & \
                             (current_state != self.tokenizer.eos_token_id)
                edit_mask = is_visible & edit_mask & initial_mask # Use initial_mask to avoid editing original prompt
                
                # Combine both masks
                update_mask = update_mask | edit_mask

            # Update current state
            current_state[update_mask] = sampled_ids[update_mask]
            
            yield current_state.clone() # Yield after each step
    
    @torch.no_grad()
    def _forward(self, model_inputs, num_steps=50, allow_edits=True) -> dict[str, Any]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer was not passed to the pipeline!")
        
        input_ids = model_inputs["input_ids"]
        all_states = list(self.diffusion_generator(input_ids, num_steps, allow_edits=allow_edits))
        final_state = all_states[-1]
    
        return {"final_state": final_state, "history": all_states}
    
    @torch.no_grad()
    def stream_generation(self, input_text, num_steps=50, allow_edits=True, max_length=None) -> Generator[str, None, None]:
        """
        Public method to stream text generation step-by-step.
        """
        # 1. Preprocess
        inputs = self.preprocess(input_text, max_length)
        input_ids = inputs["input_ids"].to(self.model.device) # type: ignore
        
        # 2. Iterate over generator
        for step_tensor in self.diffusion_generator(input_ids, num_steps, allow_edits=allow_edits):
            # Decode current state
            text = self.tokenizer.decode(step_tensor[0], skip_special_tokens=False) # type: ignore
            yield text
        
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