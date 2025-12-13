from datasets import load_dataset
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFViTForImageClassification, ViTConfig, ViTImageProcessor, Trainer, TrainingArguments
from torchvision.transforms import v2
import torch.nn as nn

# Loading dataset
animal_dataset = load_dataset("imagefolder", data_dir="../../../data/animal_images")

# Checking example structure of dataset in training data
features = animal_dataset["train"].features

# Visualization of example content of dataset
animal_dataset['train'][10]['image']

# Splitting into seperate datasets to parse onto trainer later on
train_data = animal_dataset["train"]
validation_data = animal_dataset["validation"]
test_data = animal_dataset["test"]

# Label mapping for model (label-name -> index)
id2label = {id: label for id, label in enumerate(train_data.features["label"].names)}
label2id = {label: id for id, label in id2label.items()}

# Loading processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', do_rescale = False, return_tensors = 'pt')

from torchvision import transforms 

# Get configurations from ViT processor
size = processor.size.get("height", 224)
image_mean, image_std = processor.image_mean, processor.image_std

# Normalization and augmentation transformations
transformations = {
    "train": transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ]),
    "validation": transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ]),
    "test": transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ]),
}

# Function to find the specified transformation configuration and apply it to the given example for 
def transform(examples, kind="train"):
    transform_fn = transformations.get(kind, transformations["train"])
    examples["pixel_values"] = [transform_fn(img.convert("RGB")) for img in examples["image"]]
    return examples
    
# Attaching right transformations to each dataset
train_data.set_transform(lambda examples: transform(examples, "train"))
validation_data.set_transform(lambda examples: transform(examples, "validation"))
test_data.set_transform(lambda examples: transform(examples, "test"))

import torch
from torch.utils.data import DataLoader

# Function fixes issue with data-types, as default trainer collate function is not aware how to stack the tensors from our dataset
def collate_fn(examples):
    # Stacks the pixel values of all examples into a single tensor and collects labels into a tensor
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels, "interpolate_pos_encoding": True}

# Model loading
import tensorflow as tf
from transformers import ViTForImageClassification, ViTConfig  # Config ist dieselbe Klasse

labels = animal_dataset['train'].features['label'].names
BASE_MODEL_CONF = "google/vit-base-patch16-224"

PATCH_SIZE = 8

# Label mapping and patch size are defined in config
config = ViTConfig.from_pretrained(BASE_MODEL_CONF)
config.patch_size = PATCH_SIZE
config.num_labels = len(labels)
config.id2label = id2label
config.label2id = label2id

# Updated class to force positional encoding interpolation
class ViTForImageClassificationInterpolation(ViTForImageClassification):
    def forward(
        self,
        pixel_values=None,
        labels=None,
        **kwargs,
    ):
        # Ignores multiple values for interpolation
        incoming_interpolate = kwargs.pop("interpolate_pos_encoding", None)
        # print(f"Incoming kwargs['interpolate_pos_encoding']: {incoming_interpolate}")

        use_interpolation = True
        # print(f"Effective ['interpolate_pos_encoding']: {use_interpolation}")
        
        # ViTModel.forward has argument interpolate_pos_encoding 
        return super().forward(
            pixel_values=pixel_values,
            labels=labels,
            interpolate_pos_encoding=True,
            **kwargs,
        )

# Loading model with proper label mapping + configurable patch size
model = ViTForImageClassificationInterpolation.from_pretrained(
    BASE_MODEL_CONF,
    config=config,
    ignore_mismatched_sizes=True,
)

# Freezing all layers
for name, param in model.named_parameters():
    param.requires_grad = False

# Allowing training for head and last two encoding layers
for name, param in model.named_parameters():
    if "classifier" in name:
        param.requires_grad = True

unfreeze_blocks = model.vit.encoder.layer[-2:]
# unfreeze_blocks = model.vit.encoder.layer[:]
for block in unfreeze_blocks:
    for param in block.parameters():
        param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainierbare Parameter: {trainable_params} / {total_params}")

from transformers import TrainingArguments, Trainer
import numpy as np
from transformers import EarlyStoppingCallback

# Training params
TRAINING_STRATEGY = "full_training"
BATCH_SIZE = 32
EPOCHS = 50
STEPS = 200

output_dir = f"output_vit_patch_size{PATCH_SIZE}_config_interpolation"

train_configs = {
    "full_training": TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=1,
        eval_steps=1,
        num_train_epochs=EPOCHS,
        # fp16=True,
        logging_strategy="steps",
        logging_steps=STEPS,
        learning_rate=5e-3,
        # lr_scheduler_type="linear",
        remove_unused_columns=False,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        greater_is_better=True,
        report_to="tensorboard",
        save_total_limit=3,
        seed = 123
    ),
}

import numpy as np
import evaluate
from sklearn.metrics import confusion_matrix

# Load standard evaluation metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

# Function called after completing eval strategy rule
def compute_metrics(eval_predictions):
    # Accessing model predictions
    model_calculations, true_labels = eval_predictions
    # Takes the model output with the highest value (so the most likely class to be predicted)
    model_predictions = np.argmax(model_calculations, axis=-1)
    
    # Computing all predefined metrics
    return {
        "accuracy": accuracy.compute(predictions=model_predictions, references=true_labels)["accuracy"],
        "precision": precision.compute(predictions=model_predictions, references=true_labels, average="weighted")["precision"],
        "recall": recall.compute(predictions=model_predictions, references=true_labels, average="weighted")["recall"],
        "f1": f1.compute(predictions=model_predictions, references=true_labels, average="weighted")["f1"],
    }


# Initializing and starting training
trainer = Trainer(
    model,
    train_configs.get(TRAINING_STRATEGY),
    train_dataset=train_data,
    eval_dataset=validation_data,
    data_collator=collate_fn,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
)
trainer.train()

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Evaluation of model
predictions = trainer.predict(test_data)

# Ids of True Labels
labels_true = predictions.label_ids
# Ids of predicted labels
labels_pred = np.argmax(predictions.predictions, axis=-1)

# Configs for label <-> id mapping
id2label = model.config.id2label
label2id = model.config.label2id

# Plot of confusion matrix
result_confusion = confusion_matrix(labels_true, labels_pred)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=result_confusion, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=70, values_format="d", ax=ax)

ax.set_xlabel("Predicted Labels", fontsize=15, labelpad=15)
ax.set_ylabel("True Labels", fontsize=15, labelpad=15)
ax.set_title("Confusion Matrix (Animal_Images)", fontsize=16, pad=10)

ax.tick_params(axis="x", labelsize=11, pad=5)
ax.tick_params(axis="y", labelsize=11, pad=5)
plt.tight_layout()

output_path = f"{output_dir}/confusion_matrix.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()


# Output of sklearn classification report
report = classification_report(
    labels_true,
    labels_pred,
    target_names=labels,
    output_dict=True,
)

report_df = pd.DataFrame(report).transpose()

print(f"\n\n scikit-learn report: \n{report_df}")

report_df.to_csv(f"{output_dir}/classification_report.csv", index=True)
