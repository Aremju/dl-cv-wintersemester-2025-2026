# ============================================================
# train_convnext_animals.py
# Fine-tuning ConvNeXt für Tierklassifikation (15 Klassen)
# ============================================================

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, DatasetDict
import evaluate
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. KONFIGURATION
# ============================================================

MODEL_NAME = "facebook/convnext-tiny-224"   # alternativ: "facebook/convnext-base-224"
NUM_LABELS = 15
DATA_DIR = "../../../data/animal_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training läuft auf: {device}")

# ============================================================
# 2. MODELL & PREPROCESSOR
# ============================================================

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,  # Kopf neu initialisiert (ImageNet->15 Klassen)
).to(device)

# ============================================================
# 3. DATENSATZ LADEN
# ============================================================

train_ds = load_dataset(
    "imagefolder",
    data_dir=os.path.join(DATA_DIR, "Training Data", "Training Data")
)["train"]

val_ds = load_dataset(
    "imagefolder",
    data_dir=os.path.join(DATA_DIR, "Validation Data", "Validation Data")
)["validation"]

test_ds = load_dataset(
    "imagefolder",
    data_dir=os.path.join(DATA_DIR, "Testing Data", "Testing Data")
)["test"]

dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})

print(f"📊 Datensätze geladen:")
for split in dataset:
    print(f" - {split}: {len(dataset[split])} Bilder, {len(dataset[split].features['label'].names)} Klassen")

# ============================================================
# 4. TRANSFORMATIONEN
# ============================================================

def transform(example_batch):
    images = [x.convert("RGB") for x in example_batch["image"]]
    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = example_batch["label"]
    return inputs

dataset = dataset.with_transform(transform)

# ============================================================
# 5. METRIKEN DEFINIEREN
# ============================================================

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="weighted")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="weighted")["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

# ============================================================
# 6. TRAININGSEINSTELLUNGEN
# ============================================================

args = TrainingArguments(
    output_dir="./results_convnext",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=12,
    weight_decay=0.05,
    logging_dir="./logs_convnext",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available(),  # Mixed Precision Training
    remove_unused_columns=False,     # <<< WICHTIG! verhindert den Fehler
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

# ============================================================
# 7. TRAINING & EVALUATION
# ============================================================

print("\n📦 Training ConvNeXt startet...")
train_result = trainer.train()
trainer.save_model("./convnext_finetuned_animals")
print("✅ Modell gespeichert unter ./convnext_finetuned_animals")

print("\n📊 Evaluierung auf Testdaten...")
results = trainer.evaluate(eval_dataset=dataset["test"])
print(results)

# ============================================================
# 32. TRAININGSKURVEN PLOTTEN
# ============================================================

metrics = trainer.state.log_history

def extract_metric(name):
    return [x[name] for x in metrics if name in x]

train_loss = extract_metric("loss")
eval_loss = extract_metric("eval_loss")
eval_acc = extract_metric("eval_accuracy")
eval_f1 = extract_metric("eval_f1")

os.makedirs("plots", exist_ok=True)

# Plot: Loss
plt.figure(figsize=(8,5))
plt.plot(train_loss, label="Train Loss")
plt.plot(eval_loss, label="Validation Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/loss_curve.png")
plt.show()

# Plot: Accuracy
plt.figure(figsize=(8,5))
plt.plot(eval_acc, label="Validation Accuracy", color="green")
plt.title("Validation Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/accuracy_curve.png")
plt.show()

# Plot: F1 Score
plt.figure(figsize=(8,5))
plt.plot(eval_f1, label="Validation F1", color="purple")
plt.title("Validation F1 Score over Epochs")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/f1_curve.png")
plt.show()

print("\n🏁 Fertig! Alle Plots im Ordner 'plots/' gespeichert.")
