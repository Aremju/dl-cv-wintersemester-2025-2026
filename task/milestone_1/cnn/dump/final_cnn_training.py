from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time, os, json, numpy as np
from datetime import datetime


# GPU setup
use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hyperparameters and setup
data_dir = "../../../../data/animal_images"
num_epochs = 1
batch_size = 32
input_size = (380, 380)
class_num = 15
weights_loc = None
learning_rate = 0.001
net_name = "efficientnet-b4"
epoch_to_resume_from = 0
momentum = 0.9
weight_decay = 0.0004
gamma = 0.9
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def loaddata(data_dir, batch_size, set_name, shuffle):
    """Loads and preprocesses image data for training, validation, or testing."""
    # Select correct subfolder depending on dataset split
    if set_name.lower() == "train":
        folder = os.path.join(data_dir, "train")
    elif set_name.lower() == "validation":
        folder = os.path.join(data_dir, "validation")
    elif set_name.lower() == "test":
        folder = os.path.join(data_dir, "test")
    else:
        raise ValueError(f"Unknown set_name: {set_name}")

    # Basic preprocessing and augmentations
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "validation": transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    # Create dataset and loader for specified split
    dataset = datasets.ImageFolder(folder, transform=data_transforms[set_name.lower()])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    dataset_size = len(dataset)
    print(f"Loaded '{set_name}' from: {folder} ({dataset_size} images, {len(dataset.classes)} classes)")
    return {set_name: dataloader}, dataset_size


def validate_model(model, criterion, data_dir, batch_size):
    """Runs the validation phase and calculates metrics."""
    model.eval()
    val_loaders, val_sizes = loaddata(data_dir, batch_size, "validation", shuffle=False)
    val_loss = 0.0
    all_preds, all_labels = [], []

    # Disable gradients for faster inference
    with torch.no_grad():
        for inputs, labels in val_loaders["validation"]:
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Forward pass
            outputs = model(inputs)
            loss_val = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)

            # Accumulate loss and predictions
            val_loss += loss_val.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate mean metrics
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    metrics = {
        "loss": val_loss / val_sizes,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0)
    }
    return metrics


def create_log_entry(epoch, train_loss, train_acc, lr, val_metrics):
    """Builds structured log entries for training and validation."""
    train_log = {
        "epoch": epoch + 1,
        "loss": round(train_loss, 4),
        "accuracy": round(train_acc.item(), 4),
        "learning_rate": lr
    }
    val_log = {
        "epoch": epoch + 1,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"]
    }
    return train_log, val_log


def save_metrics(train_logs, val_logs, test_metrics=None):
    """Saves training, validation, and test metrics as JSON files."""
    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

    # Store training and validation history
    with open(os.path.join(os.getcwd(), f"logs/train_logs_{timestamp}.json"), "w") as f:
        json.dump(train_logs, f, indent=4)
    with open(os.path.join(os.getcwd(), f"logs/val_logs_{timestamp}.json"), "w") as f:
        json.dump(val_logs, f, indent=4)

    # Store test metrics if available
    if test_metrics:
        with open(os.path.join(os.getcwd(), f"logs/model_report_{timestamp}.json"), "w") as f:
            json.dump(test_metrics, f, indent=4)


def save_best_model(model, model_wts, net_name, timestamp):
    """Saves the best model weights after training."""
    model.load_state_dict(model_wts)
    model_out_path = os.path.join(os.getcwd(), f"{net_name}_best_{timestamp}.pth")
    torch.save(model, model_out_path)
    print(f"Model saved to {model_out_path}")


def train_model(model_ft, criterion, optimizer, scheduler, num_epochs=50):
    """Trains the model, validates each epoch, and saves logs and best weights."""
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    train_logs, val_logs = [], []
    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), "runs"))

    for epoch in range(epoch_to_resume_from, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # Enable training mode
        model_ft.train()
        dset_loaders, dset_sizes = loaddata(data_dir, batch_size, "train", shuffle=True)
        running_loss = 0.0
        running_corrects = 0
        count = 0

        # Train over all batches
        for inputs, labels in dset_loaders["train"]:
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            count += 1

            # Display intermediate batch stats
            if count % batch_size == 0:
                batch_acc = (running_corrects.double() / (count * batch_size)).item()
                print(f"[Batch {count}] Loss: {loss.item():.4f} | Train Acc: {batch_acc:.4f}")

        # Epoch-level results
        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes

        # Validation phase
        val_metrics = validate_model(model_ft, criterion, data_dir, batch_size)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Output epoch metrics
        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"Prec: {val_metrics['precision']:.4f} | "
              f"Rec: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f} | LR: {current_lr:.6f}\n")

        # Log metrics for export
        train_log, val_log = create_log_entry(epoch, epoch_loss, epoch_acc, current_lr, val_metrics)
        train_logs.append(train_log)
        val_logs.append(val_log)

        # Save best model if validation improves
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_model_wts = model_ft.state_dict()

    # Save logs and model
    save_metrics(train_logs, val_logs)
    save_best_model(model_ft, best_model_wts, net_name, timestamp)

    writer.close()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    return train_logs, val_logs, best_model_wts


def test_model(model, criterion):
    """Evaluates the trained model on the test set and saves final metrics."""
    model.eval()
    dset_loaders, dset_sizes = loaddata(data_dir, batch_size, "test", shuffle=False)
    all_preds, all_labels = [], []
    running_loss = 0.0

    # Disable gradient computation for test
    with torch.no_grad():
        for inputs, labels in dset_loaders["test"]:
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute overall test metrics
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    test_loss = running_loss / dset_sizes
    test_metrics = {
        "timestamp": timestamp,
        "model_name": net_name,
        "test_metrics": {
            "loss": test_loss,
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0)
        }
    }

    # Save to JSON
    save_metrics([], [], test_metrics)
    print("Model report saved for test metrics.")
    return test_metrics


def run():
    """Initializes model, optimizer, scheduler, and runs training and testing."""
    # Load pretrained model or existing weights
    if weights_loc:
        model_ft = torch.load(weights_loc)
    else:
        model_ft = EfficientNet.from_pretrained(net_name)

    # Adjust final layer to dataset class count
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, class_num)

    # Move model to GPU if available
    if use_gpu:
        model_ft = model_ft.cuda()

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss().cuda() if use_gpu else nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Train model and run final evaluation
    train_logs, val_logs, best_model_wts = train_model(model_ft, criterion, optimizer, scheduler, num_epochs=num_epochs)
    print("-" * 10)
    print("Testing best model on validation set...")
    model_ft.load_state_dict(best_model_wts)
    test_model(model_ft, criterion)


if __name__ == "__main__":
    """Runs the complete training and evaluation pipeline."""
    print(f"Dataset: {data_dir} | Epochs: {num_epochs} | Batch size: {batch_size} | Classes: {class_num}")
    run()
