from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import time
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np


# GPU Access
use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Hyperparameters
data_dir = "../../../../data/animal_images"
num_epochs = 1
batch_size = 32
class_num = 15
weights_location = None
weight_decay = 0.0004
learning_rate = 0.001
net_name = "efficientnet-b4"
epoch_to_resume_from = 0
momentum = 0.9


# Loads all split sets of the original dataset
def loaddata(data_dir, batch_size, set_name, shuffle):

    # Define dataset folders for all splits
    if set_name.lower() == "train":
        folder = os.path.join(data_dir, "train")
    elif set_name.lower() == "validation":
        folder = os.path.join(data_dir, "validation")
    elif set_name.lower() == "test":
        folder = os.path.join(data_dir, "test")
    else:
        raise ValueError(f"Unknown set_name: {set_name}")

    # Normalization and augmentation parameters — based on ImageNet training setup
    # Training set has heavy augmentations, validation/test use consistent preprocessing
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(input_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "validation": transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    # Load datasets for all splits
    split_datasets = {
        x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x])
        for x in ["train", "validation", "test"]
    }

    # Create DataLoaders for each split
    dataloaders = {
        x: torch.utils.data.DataLoader(
            split_datasets[x],
            batch_size=batch_size,
            shuffle=(x == "train") and shuffle,  # only shuffle training data
            num_workers=0,
        )
        for x in ["train", "validation", "test"]
    }

    # Dataset size summary
    dataset_sizes = {x: len(split_datasets[x]) for x in ["train", "validation", "test"]}
    for x in ["train", "validation", "test"]:
        print(f"Loaded '{x}' from: {data_dir}/{x} ({dataset_sizes[x]} images, {len(split_datasets[x].classes)} classes)")

    return dataloaders, dataset_sizes


# Performs validation during training
def validate_model(model, criterion, data_dir, batch_size, use_gpu):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []

    # Load validation set
    val_loaders, val_sizes = loaddata(data_dir, batch_size, "val", shuffle=False)

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in val_loaders["val"]:
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Forward pass
            outputs = model(inputs)
            loss_val = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Accumulate loss and predictions
            val_loss += loss_val.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Convert collected results to arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Calculate metrics
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    epoch_val_loss = val_loss / val_sizes["val"]

    # Return results as dict
    metrics = {
        "loss": epoch_val_loss,
        "accuracy": val_accuracy,
        "precision": val_precision,
        "recall": val_recall,
        "f1": val_f1
    }

    return metrics


# Logs most important metrics to console and TensorBoard
def log_epoch_metrics(writer, epoch, train_loss, train_acc, val_loss, val_acc, current_lr):
    print(f"[{phase}] "
          f"Loss: {metrics['loss']:.4f} | "
          f"Acc: {metrics['accuracy']:.4f} | "
          f"P: {metrics['precision']:.4f} | "
          f"R: {metrics['recall']:.4f} | "
          f"F1: {metrics['f1']:.4f}")

    # TensorBoard logging if writer is provided
    if writer:
        writer.add_scalar(f"Loss/{phase}", metrics["loss"], epoch)
        writer.add_scalar(f"Accuracy/{phase}", metrics["accuracy"], epoch)
        writer.add_scalar(f"Precision/{phase}", metrics["precision"], epoch)
        writer.add_scalar(f"Recall/{phase}", metrics["recall"], epoch)
        writer.add_scalar(f"F1/{phase}", metrics["f1"], epoch)
        if current_lr is not None:
            writer.add_scalar("LearningRate", current_lr, epoch)


# Saves the best model weights to disk
def save_best_model(model, best_model_wts, data_dir, net_name):
    os.makedirs(os.path.join(data_dir, "model"), exist_ok=True)
    model.load_state_dict(best_model_wts)
    model_out_path = os.path.join(data_dir, "model", f"{net_name}.pth")
    torch.save(model, model_out_path)
    print(f"Model saved to {model_out_path}")


# Trains the model and evaluates each epoch
def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50):
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    train_loss_list = []

    # TensorBoard writer setup
    writer = SummaryWriter(log_dir=os.path.join(data_dir, "runs"))

    for epoch in range(epoch_to_resume_from, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # Set model to training mode
        model_ft.train(True)
        dset_loaders, dset_sizes = loaddata(data_dir, batch_size, "train", shuffle=True)

        running_loss = 0.0
        running_corrects = 0
        count = 0

        # Loop through batches
        for inputs, labels in dset_loaders["train"]:
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Forward pass + Backpropagation + Optimization
            optimizer.zero_grad()
            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            count += 1

            # Print every few batches for progress
            if count % batch_size == 0:
                print(f"[Batch {count}] Loss: {loss.item():.4f}")

        # Epoch-level training statistics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        train_loss = running_loss / dset_sizes["train"]

        train_metrics = {
            "loss": train_loss,
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        }

        # Scheduler update
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr:.6f}")

        # Validation phase after each epoch
        val_metrics = validate_model(model, criterion, data_dir, batch_size, use_gpu)
        
        # Logging results
        log_epoch_metrics(writer, epoch, "Train", train_metrics, current_lr)
        log_epoch_metrics(writer, epoch, "Val", val_metrics)
        print()
        train_loss_list.append(train_loss)
        
        # Save best model if accuracy improved
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_model_wts = copy.deepcopy(model.state_dict())

    # Final training summary
    writer.close()
    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation accuracy: {best_acc:.4f}")

    # Save best model weights
    save_best_model(model, best_model_wts, data_dir, net_name)

    return train_loss_list, best_model_wts


# Evaluates trained model on test set
def test_model(model, criterion):
    model.eval()
    dset_loaders, dset_sizes = loaddata(data_dir, batch_size, "val", shuffle=False)
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dset_loaders["val"]:
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Compute test predictions and loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    print(f"Test Loss: {running_loss / dset_sizes:.4f} Acc: {running_corrects.double() / dset_sizes:.4f}")


# Main function running training and testing consecutively
def run():
    if weights_location:
        model_pre_trained = torch.load(weights_location)
    else:
        model_pre_trained = EfficientNet.from_pretrained(net_name)

    # Adjust final layer to match dataset classes
    model_features = model_pre_trained._fc.in_features
    model_pre_trained._fc = nn.Linear(model_features, class_num)

    # Move model to GPU if available
    if use_gpu:
        model_pre_trained = model_pre_trained.cuda()

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss().cuda() if use_gpu else nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model_pre_trained.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Cosine annealing learning rate scheduler
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # Train model
    train_loss, best_model_weights = train_model(
        model_pre_trained,
        criterion,
        optimizer,
        cosine_scheduler,
        num_epochs=num_epochs,
    )

    # Test model on best weights
    print("-" * 10)
    print("Testing best model on test set")
    model_pre_trained.load_state_dict(best_model_wts)
    test_model(model_pre_trained, criterion)


# Entry point
if __name__ == "__main__":
    print(f"Dataset: {data_dir} | Epochs: {num_epochs} | Batch size: {batch_size} | Classes: {class_num}")
    run()
