from datasets import load_dataset
from tasks.Task import Task

class EverydayConversationsTask(Task):
    def load_dataset(self):
        dataset = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")
            
        return dataset