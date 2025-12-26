from datasets import load_dataset

train_dataset = load_dataset("roneneldan/TinyStories", split="train[:1]")
# eval_dataset = load_dataset("roneneldan/TinyStories", split="validation")
print(train_dataset[0])
print(len(train_dataset))
print(len(train_dataset[0]['text']))   