from datasets import Dataset

class Task:
    def __init__(self, seed=42) -> None:
        self.seed = seed

    def load_dataset(self) -> Dataset:
        """Loads the dataset and normalizes it to have a 'messages' column."""
        raise NotImplementedError("Each task must implement load_dataset")

    def name(self) -> str:
        return self.__class__.__name__