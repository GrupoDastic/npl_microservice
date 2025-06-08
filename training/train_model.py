import os
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
import json


class CustomLabelEncoder:
    def __init__(self):
        self.label2id = {}
        self.id2label = {}

    def fit_transform(self, labels):
        for label in labels:
            if label not in self.label2id:
                id = len(self.label2id)
                self.label2id[label] = id
                self.id2label[id] = label
        return [self.label2id[label] for label in labels]

    def get_mappings(self):
        return self.label2id, self.id2label


script_dir = os.path.dirname(os.path.abspath(__file__))          # /training
dataset_path = os.path.join(script_dir, "dataset.csv")           # /training/dataset.csv
model_dir = os.path.join(script_dir, "..", "model")              # /model (1 nivel arriba)
log_dir = os.path.join(script_dir, "..", "logs")                 # /logs

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

dataset = load_dataset("csv", data_files=dataset_path)["train"]

label_encoder = CustomLabelEncoder()


def process_labels(example):
    try:
        label = example["label"]
        return {"labels": label_encoder.fit_transform([label])[0]}
    except KeyError:
        raise ValueError("Missing 'label' field in the dataset")


dataset = dataset.map(process_labels, remove_columns=["label"])

label2id, id2label = label_encoder.get_mappings()
with open(os.path.join(model_dir, "label2id.json"), "w") as f:
    json.dump(label2id, f)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


def tokenize(example):
    try:
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    except KeyError:
        raise ValueError("Missing 'text' field in the dataset")


tokenized_dataset = dataset.filter(
    lambda example: "text" in example and isinstance(example["text"], str) and example["text"].strip()
)
tokenized_dataset = tokenized_dataset.map(tokenize, batched=True)

# 8. Prepare model and training
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir=model_dir,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    save_strategy="epoch",
    logging_dir=log_dir,
    logging_steps=10,
    load_best_model_at_end=False
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = (predictions == labels).mean()
    return {"accuracy": acc}


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# 9. Train and save
trainer.train()
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

print(f"\n✅ Entrenamiento completado. Modelo guardado en: {model_dir}")
