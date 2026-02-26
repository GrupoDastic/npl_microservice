"""
Script de entrenamiento mejorado para clasificación de intenciones - Parqueadero NLP

Mejoras:
1. BETO (bert-base-spanish) en vez de bert-multilingual
2. Train/eval split 80/20 estratificado
3. Early stopping para evitar overfitting
4. Learning rate optimizado con warmup
5. Data augmentation con sinónimos del dominio
6. Label encoding seguro (fit separado de transform)
7. Métricas completas (accuracy, f1, precision, recall)
8. max_length reducido a 64 (más eficiente)
"""

import os
import json
import random
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ============================================================
# CONFIGURACIÓN
# ============================================================
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EVAL_SPLIT = 0.2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# RUTAS (adaptadas a tu estructura)
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))  # training/
dataset_path = os.path.join(script_dir, "dataset.csv")
model_dir = os.path.join(script_dir, "..", "model")
log_dir = os.path.join(script_dir, "..", "logs")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset no encontrado en: {dataset_path}")

# ============================================================
# DATA AUGMENTATION - Sinónimos del dominio
# ============================================================
SYNONYM_MAP = {
    "parqueadero": ["parqueo", "estacionamiento", "parking", "espacio"],
    "parqueo": ["parqueadero", "estacionamiento", "lugar"],
    "estacionar": ["parquear", "aparcar"],
    "parquear": ["estacionar", "aparcar"],
    "libre": ["disponible", "vacío", "desocupado"],
    "disponible": ["libre", "vacío"],
    "ocupado": ["no disponible", "lleno", "en uso"],
    "zona": ["área", "sector"],
    "franja": ["sección", "línea", "fila"],
}


def augment_text(text, probability=0.3):
    """Genera una variación del texto reemplazando sinónimos aleatoriamente."""
    words = text.split()
    new_words = []
    changed = False
    for word in words:
        word_lower = word.lower().strip("¿?.,!¡")
        if word_lower in SYNONYM_MAP and random.random() < probability:
            synonym = random.choice(SYNONYM_MAP[word_lower])
            if word[0].isupper():
                synonym = synonym.capitalize()
            new_words.append(synonym)
            changed = True
        else:
            new_words.append(word)
    if changed:
        return " ".join(new_words)
    return None


# ============================================================
# LABEL ENCODING (seguro: fit primero, transform después)
# ============================================================
class LabelEncoder:
    def __init__(self):
        self.label2id = {}
        self.id2label = {}

    def fit(self, labels):
        unique_labels = sorted(set(labels))  # Ordenado para reproducibilidad
        for idx, label in enumerate(unique_labels):
            self.label2id[label] = idx
            self.id2label[idx] = label
        print(f"  {len(unique_labels)} clases encontradas: {unique_labels}")

    def transform(self, label):
        return self.label2id[label]


# ============================================================
# CARGAR Y PREPARAR DATOS
# ============================================================
print("Cargando dataset...")
dataset = load_dataset("csv", data_files=dataset_path)["train"]

# Eliminar duplicados
seen = set()
unique_indices = []
for i, example in enumerate(dataset):
    key = (example["text"], example["label"])
    if key not in seen:
        seen.add(key)
        unique_indices.append(i)

dataset = dataset.select(unique_indices)
print(f"Ejemplos unicos: {len(dataset)}")

# Fit del encoder
label_encoder = LabelEncoder()
all_labels = [ex["label"] for ex in dataset]
label_encoder.fit(all_labels)

# Data augmentation
print("Aplicando data augmentation...")
augmented_texts = []
augmented_labels = []
for example in dataset:
    augmented = augment_text(example["text"])
    if augmented and augmented != example["text"]:
        augmented_texts.append(augmented)
        augmented_labels.append(example["label"])

if augmented_texts:
    aug_dataset = Dataset.from_dict({"text": augmented_texts, "label": augmented_labels})
    dataset = concatenate_datasets([dataset, aug_dataset])
    print(f"Ejemplos despues de augmentation: {len(dataset)}")


# Codificar labels
def encode_labels(example):
    return {"labels": label_encoder.transform(example["label"])}


dataset = dataset.map(encode_labels, remove_columns=["label"])

# Guardar mappings
label2id = label_encoder.label2id
id2label = {str(k): v for k, v in label_encoder.id2label.items()}

with open(os.path.join(model_dir, "label2id.json"), "w") as f:
    json.dump(label2id, f, indent=2)

# ============================================================
# TOKENIZACIÓN
# ============================================================
print(f"Cargando tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding=False,  # DataCollator hace padding dinámico
        max_length=MAX_LENGTH,
    )


tokenized_dataset = dataset.map(tokenize, batched=True)

# ============================================================
# TRAIN / EVAL SPLIT (estratificado)
# ============================================================
print(f"Dividiendo en train/eval ({int((1 - EVAL_SPLIT) * 100)}/{int(EVAL_SPLIT * 100)})...")

# Convertir labels a ClassLabel para poder estratificar
from datasets import ClassLabel
class_names = [label_encoder.id2label[i] for i in range(len(label_encoder.id2label))]
tokenized_dataset = tokenized_dataset.cast_column("labels", ClassLabel(names=class_names))

split = tokenized_dataset.train_test_split(
    test_size=EVAL_SPLIT,
    seed=SEED,
    stratify_by_column="labels",
)
train_dataset = split["train"].cast_column("labels", datasets.Value("int64"))
eval_dataset = split["test"].cast_column("labels", datasets.Value("int64"))

print(f"   Train: {len(train_dataset)} ejemplos")
print(f"   Eval:  {len(eval_dataset)} ejemplos")

# ============================================================
# MODELO
# ============================================================
print(f"Cargando modelo: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)


# ============================================================
# MÉTRICAS
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "precision_weighted": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels, predictions, average="weighted", zero_division=0),
    }


# ============================================================
# ENTRENAMIENTO
# ============================================================
training_args = TrainingArguments(
    output_dir=model_dir,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    seed=SEED,
    data_seed=SEED,
    save_total_limit=2,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
    ],
)

print("\nIniciando entrenamiento...")
train_result = trainer.train()

# ============================================================
# RESULTADOS FINALES
# ============================================================
print("\nEvaluacion final en conjunto de eval:")
metrics = trainer.evaluate()
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"   {key}: {value:.4f}")

trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

print(f"\nEntrenamiento completado!")
print(f"   Modelo guardado en: {model_dir}")
print(f"   Accuracy: {metrics.get('eval_accuracy', 'N/A')}")
print(f"   F1 (weighted): {metrics.get('eval_f1_weighted', 'N/A')}")