import os
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

class CustomLabelEncoder:
    def __init__(self):
        self.label2id = {}
        self.id2label = {}

    def fit(self, labels):
        for label in labels:
            if label not in self.label2id:
                idx = len(self.label2id)
                self.label2id[label] = idx
                self.id2label[idx] = label

    def transform(self, labels):
        return [self.label2id[label] for label in labels]

    def get_mappings(self):
        return self.label2id, self.id2label

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "dataset.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

dataset = load_dataset("csv", data_files=dataset_path)["train"]

label_encoder = CustomLabelEncoder()
all_labels = [ex["label"] for ex in dataset]
label_encoder.fit(all_labels)
encoded_labels = label_encoder.transform(all_labels)

dataset = dataset.remove_columns(["label"])
dataset = dataset.add_column("labels", encoded_labels)

label2id, id2label = label_encoder.get_mappings()
os.makedirs(os.path.join(script_dir, "../model"), exist_ok=True)
with open(os.path.join(script_dir, "../model/label2id.json"), "w") as f:
    json.dump(label2id, f)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    output_dir=os.path.join(script_dir, "../model"),
    per_device_train_batch_size=8,
    num_train_epochs=5,
    save_steps=500,
    logging_dir=os.path.join(script_dir, "../logs"),
    logging_steps=10
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

trainer.train()
trainer.save_model(os.path.join(script_dir, "../model"))
tokenizer.save_pretrained(os.path.join(script_dir, "../model"))
