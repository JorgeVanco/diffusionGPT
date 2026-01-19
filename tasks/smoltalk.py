from datasets import load_dataset
from tasks.Task import Task

class SmolTalkTask(Task):
    def load_dataset(self):
        # Load the dataset
        dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
        
        return dataset