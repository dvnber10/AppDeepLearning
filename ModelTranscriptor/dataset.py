from datasets import load_dataset

dataset = load_dataset("wmt14", "es-en")

# Un ejemplo
print(dataset["train"][0])