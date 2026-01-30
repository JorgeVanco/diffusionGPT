from transformers import BatchEncoding, Pipeline
import torch
from typing import Any, Generator

class TextDiffusionPipeline(Pipeline):
    def _sanitize_parameters(
        self,
        num_steps: int = 50,
        allow_edits: bool = True,
        use_confidence: bool = False,
        stop_token: None = None,
        **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        forward_kwargs = {
            "num_steps": num_steps,
            "allow_edits": allow_edits,
            "use_confidence": use_confidence,
            "stop_token": stop_token
        }
        
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
            
        tokenized_text = self.tokenizer.encode(input_text)
        
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
    def diffusion_generator(
        self,
        input_ids: torch.Tensor,
        num_steps: int,
        allow_edits: bool = True,
        use_confidence: bool = False
    ) -> Generator[torch.Tensor, None, None]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer was not passed to the pipeline!")
        
        current_state: torch.Tensor = input_ids.clone()
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
            dist = torch.distributions.Categorical(probs)
            sampled_ids = dist.sample()
            
            # Calculate Unmasking Probability (Equation 7 https://arxiv.org/pdf/2406.07524)
            # P(unmask | masked) = (alpha_s - alpha_t) / (1 - alpha_t)
            # mapping: alpha_t = (1 - t_current), alpha_s = (1 - t_next)
            # resulting simplified formula: (t_current - t_next) / t_current
            if step < num_steps - 1:
                unmasking_prob = (t_current - t_next) / t_current
            else:
                unmasking_prob = 1.0 # Force unmask at the end
            
            remasking_mask: torch.Tensor = (current_state == self.tokenizer.mask_token_id) | \
                             (current_state == self.tokenizer.pad_token_id) # type: ignore
            
            if use_confidence:
                # Get the confidence (probability) of the tokens we just sampled
                sample_probs = probs.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
                
                # Determine how many tokens to unmask this step
                if step < num_steps - 1:
                    num_masked = remasking_mask.sum(dim=1, keepdim=True)
                    num_to_unmask = (num_masked.float() * unmasking_prob).ceil().long()
                else:
                    num_to_unmask = remasking_mask.sum(dim=1, keepdim=True)
                
                # Select Top-K most confident tokens
                # Set confidence of already visible tokens to -inf so they aren't picked
                candidate_confidences = sample_probs.clone()
                candidate_confidences[~remasking_mask] = -float('inf')
                
                unmasking_mask = torch.zeros_like(remasking_mask, dtype=torch.bool)
                
                max_k = num_to_unmask.max().item()
                if max_k > 0:
                    _, top_indices = candidate_confidences.topk(k=max_k, dim=1) # type: ignore
                    range_tensor = torch.arange(max_k, device=current_state.device).unsqueeze(0)
                    mask_k = range_tensor < num_to_unmask
                    unmasking_mask.scatter_(1, top_indices, mask_k)

            else:
                # Random Unmasking
                unmasking_mask = torch.rand_like(current_state, dtype=torch.float) < unmasking_prob
                        
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
    def _forward(
        self,
        model_inputs,
        num_steps: int = 50,
        allow_edits: bool = True,
        use_confidence: bool = False,
        stop_token: str | None = None
    ) -> dict[str, Any]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer was not passed to the pipeline!")
        
        input_ids = model_inputs["input_ids"]
        all_states = list(self.diffusion_generator(input_ids=input_ids, num_steps=num_steps, allow_edits=allow_edits, use_confidence=use_confidence))
        final_state = all_states[-1]
    
        return {"final_state": final_state, "history": all_states}
    
    @torch.no_grad()
    def stream_generation(
        self,
        input_text: str,
        num_steps: int = 50,
        allow_edits: bool = True,
        use_confidence: bool = False,
        max_length: int | None = None, 
        stop_token: str | None = None
    ) -> Generator[str, None, None]:
        """
        Public method to stream text generation step-by-step.
        """
        # 1. Preprocess
        inputs = self.preprocess(input_text, max_length)
        input_ids = inputs["input_ids"].to(self.model.device) # type: ignore
        
        # 2. Iterate over generator
        for step_tensor in self.diffusion_generator(input_ids=input_ids, num_steps=num_steps, allow_edits=allow_edits, use_confidence=use_confidence):
            # Decode current state
            text = self.tokenizer.decode(step_tensor[0], skip_special_tokens=False) # type: ignore
            yield text
            
        if stop_token is not None and stop_token in text[len(input_text):]:
            text = input_text + text[len(input_text):].split(stop_token)[0]
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
        
    @torch.no_grad()
    def block_diffusion_generator(
        self, input_ids: torch.Tensor,
        block_size: int,
        max_length: int,
        num_steps: int,
        allow_edits: bool = True,
        use_confidence: bool = False,
        stop_token: str | None = None
    ) -> Generator[torch.Tensor, None, None]:
        """
        Generator that yields the diffusion states block-by-block.
        Args:
            input_ids (torch.Tensor): Initial input IDs with context.
            block_size (int): Number of tokens to generate in each block.
            max_length (int): Max length of the generated text.
            num_steps (int): Number of diffusion steps per block.
            allow_edits (bool): Whether to allow edits to existing tokens.
            use_confidence (bool): Whether to use confidence-based unmasking.
            stop_token (str | None): Token at which to stop generation early.
        Yields:
            torch.Tensor: The current state of the full sequence after each diffusion step.
        """
        assert num_steps > 0, "num_steps must be greater than 0"
        if self.tokenizer is None:
            raise ValueError("Tokenizer was not passed to the pipeline!")
        
        max_seq_length = self.model.config.seq_length if hasattr(self.model.config, "seq_length") else 512
        stop_token_id = self.tokenizer.convert_tokens_to_ids(stop_token) if stop_token is not None else None
        
        assert block_size > 0 and block_size <= max_seq_length, f"block_size must be in (0, {max_seq_length}]"
        
        full_sequence = input_ids.clone()
        current_length = input_ids.shape[1]
        while current_length < max_length:
            remaining = max_length - current_length
            this_block_len = min(block_size, remaining)
            if this_block_len <= 0: break
            
            # Append MASK tokens for the new block
            mask_block = torch.full(
                (1, this_block_len), 
                self.tokenizer.mask_token_id, # type: ignore
                dtype=torch.long, 
                device=self.model.device
            )
            
            # Combine Context + New Masks
            input_ids = torch.cat([full_sequence[:, -(max_seq_length - this_block_len):], mask_block], dim=1)
            
            for step_tensor in self.diffusion_generator(
                input_ids, 
                num_steps=num_steps, 
                allow_edits=allow_edits,
                use_confidence=use_confidence
            ):
                current_generated_tokens = step_tensor[:, -this_block_len:]
                yield torch.cat([full_sequence, current_generated_tokens], dim=1)
                
            
            if stop_token_id is not None and stop_token_id in current_generated_tokens:
                # Stop if EOS is generated
                eos_index = (current_generated_tokens == stop_token_id).nonzero(as_tuple=True)[1] # type: ignore
                current_generated_tokens = current_generated_tokens[:, :eos_index[0]]
                yield torch.cat([full_sequence, current_generated_tokens], dim=1)
                break

            # Update full sequence and current length
            full_sequence = torch.cat([full_sequence, current_generated_tokens], dim=1)
            current_length = full_sequence.shape[1]
        
    
    @torch.no_grad()
    def semi_autoregressive_generate(
        self, 
        input_text: str, 
        block_size: int = 64, 
        max_length: int = 256, 
        num_steps: int = 50,
        allow_edits: bool = True,
        use_confidence: bool = False
    ) -> dict[str, Any]:
        """
        Semi-Autoregressive Generation:
        Generates text in blocks using the diffusion model.
        Each block is generated by appending MASK tokens to the current context
        and running the diffusion process on the combined sequence.
        Args:
            input_text (str): The initial prompt text.
            block_size (int): Number of tokens to generate in each block.
            max_length (int): Max length of the generated text.
            num_steps (int): Number of diffusion steps per block.
            allow_edits (bool): Whether to allow edits to existing tokens.
            use_confidence (bool): Whether to use confidence-based unmasking.
        Returns:
            dict[str, Any]: A dictionary containing the decoded texts, generation history, and final token IDs.
        """
        if self.tokenizer is None: raise ValueError("No tokenizer")
        
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device) # type: ignore
        all_states = list(self.block_diffusion_generator(input_ids, block_size, max_length, num_steps, allow_edits, use_confidence=use_confidence))
        final_state = all_states[-1]
        return {
            "decoded_texts": self.tokenizer.batch_decode(final_state, skip_special_tokens=False),
            "history": all_states,
            "final_ids": final_state
        }
    
    @torch.no_grad()
    def stream_semi_autoregressive_generate(
        self, 
        input_text: str, 
        block_size: int = 64, 
        max_length: int = 256, 
        num_steps: int = 50,
        allow_edits: bool = True,
        use_confidence: bool = False,
        stop_token: str | None = None
    ) -> Generator[str, None, None]:
        """
        Streams the generation process block-by-block.
        Yields the full decoded text at every diffusion step of every block.
        Args:
            input_text (str): The initial prompt text.
            block_size (int): Number of tokens to generate in each block.
            max_length (int): Max length of the generated text.
            num_steps (int): Number of diffusion steps per block.
            allow_edits (bool): Whether to allow edits to existing tokens.
            use_confidence (bool): Whether to use confidence-based unmasking.
            stop_token (None): Token at which to stop generation early.
        Yields:
            str: The current generated text after each diffusion step.
        """
        if self.tokenizer is None: raise ValueError("No tokenizer")
        
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device) # type: ignore
        
        for step_tensor in self.block_diffusion_generator(input_ids, block_size, max_length, num_steps, allow_edits, use_confidence=use_confidence, stop_token=stop_token):
            # Decode current state
            yield self.tokenizer.decode(step_tensor[0], skip_special_tokens=False) # type: ignore