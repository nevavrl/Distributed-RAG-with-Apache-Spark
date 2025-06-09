from datasets import load_dataset

dataset = load_dataset("wikipedia", language="en", date="20220301")
print(dataset["train"][20])
