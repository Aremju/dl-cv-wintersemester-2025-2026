from datasets import load_dataset
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from torchvision.transforms import v2

train_ds = load_dataset("imagefolder", data_dir="../../../data/animal_images/Training Data/Training Data")
val_ds = load_dataset("imagefolder", data_dir="../../../data/animal_images/Validation Data/Validation Data")

labels = train_ds['train'].features['label'].names
print(labels)

"""train_ds = train_ds["train"]
val_ds = val_ds["train"]


model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', do_rescale = False, return_tensors = 'pt')

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = v2.Normalize(mean=image_mean, std=image_std)

train_transform = v2.Compose([
      v2.Resize((processor.size["height"], processor.size["width"])),
      v2.RandomHorizontalFlip(0.4),
      v2.RandomVerticalFlip(0.1),
      v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 90))], p=0.5),
      v2.RandomApply(transforms=[v2.ColorJitter(brightness=.3, hue=.1)], p=0.3),
      v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=(5, 9))], p=0.3),
      v2.ToTensor(),
      normalize
      #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
 ])

def train_transforms(examples):
    examples['pixel_values'] = [train_transform(image.convert("RGB")) for image in examples['image']]
    return examples
 
train_ds.set_transform(train_transforms)
val_ds.set_transform(train_transforms)

metric_name = "accuracy"

# Define Train Parameters
args = TrainingArguments(
    f"breed-classification",
    use_cpu = False,
    eval_strategy="steps",
    logging_steps = 100,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=15,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
)

# Train
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    # compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()"""