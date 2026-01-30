from datasets import load_dataset, concatenate_datasets
from tasks.Task import Task

class NemotronTask(Task):
    def load_dataset(self):
        # Nemotron has two subsets: 'chat_if' and 'structured_outputs'
        # We can load both and concatenate, or just one.
        
        ds_chat = load_dataset("nvidia/Nemotron-Instruction-Following-Chat-v1", split="chat_if[:20000]")
        # ds_struct = load_dataset("nvidia/Nemotron-Instruction-Following-Chat-v1", split="structured_outputs")
        
        # Concatenate if you want both
        # dataset = concatenate_datasets([ds_chat, ds_struct])
        dataset = ds_chat
        
        # Function to standardise messages: keep ONLY 'role' and 'content'
        def format_messages(example):
            clean_messages = []
            for msg in example["messages"]:
                clean_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                    # Drop 'reasoning_content'
                })
            return {"messages": clean_messages}

        # Apply the formatting
        dataset = dataset.map(format_messages, desc="Standardizing Nemotron schema")
            
        return dataset