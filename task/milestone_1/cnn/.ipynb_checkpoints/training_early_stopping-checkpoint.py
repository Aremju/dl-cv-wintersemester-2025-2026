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
data_dir = "../../../data/animal_images"
num_epochs = 20
batch_size = 16
input_size = (380, 380)
class_num = 15
weights_loc = None
learning_rate = 2e-4
net_name = "efficientnet-b7"
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
            # transforms.RandomVerticalFlip(),
            # transforms.RandomResizedCrop(input_size),
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
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

class EarlyStopping:
    """Early stops the training when the monitored metric has stopped improving"""
    def __init__(self, patience=3, delta=0.0, mode="min", metric_name="metric"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.metric_name = metric_name

        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

        self.best = None
        self.counter = 0
        self.early_stop = False

    def step(self, value):
        """Called in a training step to check if early stopping is required"""
        if self.best is None:
            self.best = value
            return

        # Calculate if improvement is there
        improvement = (
            (self.mode == "min" and value < self.best - self.delta) or
            (self.mode == "max" and value > self.best + self.delta)
        )

        if improvement:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping: No improvement in {self.metric_name}. Patience Counter {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print(f"\nEarly stopping on {self.metric_name}!!!\n")
                self.early_stop = True


def train_model(model_ft, criterion, optimizer, scheduler, num_epochs=50):
    """Trains the model, validates each epoch, and saves logs and best weights."""
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    train_logs, val_logs = [], []

    # Initializing early stopping
    monitor_metric = "val_accuracy"
    # monitor_metric = "batch_acc"
    patience = 3
    
    early_stopper = EarlyStopping(
        patience=patience,
        delta=1.000,
        mode="max",
        metric_name=monitor_metric
    )
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), "runs", timestamp))
    print(f"TensorBoard logs will be saved to: ./runs/{timestamp}")

    for epoch in range(epoch_to_resume_from, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # Enable training mode
        model_ft.train()
        dset_loaders, dset_sizes = loaddata(data_dir, batch_size, "train", shuffle=True)
        running_loss = 0.0
        running_corrects = 0
        batch_counter = 0

        # Train over all batches
        for inputs, labels in dset_loaders["train"]:
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)

            # Backward pass + optimize
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            batch_counter += 1

            # Display collective batch stats every few batches
            if batch_counter % 32 == 0:
                batch_acc = (running_corrects.double() / (batch_counter * batch_size)).item()
                print(f"[Batch {batch_counter}] Loss: {loss.item():.4f} | Train Acc: {batch_acc:.4f}")

                # Evaluation of early stopping in batch training phase
                """early_stopper.step(batch_acc)
            
                if early_stopper.early_stop:
                    print(f"Training stopped early. Best {monitor_metric}: {early_stopper.best}")
                    best_model_wts = model_ft.state_dict()
                    break"""

        # Calculate epoch statistics
        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes

        # Validation phase
        model_ft.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        # Load validation data
        val_loaders, val_sizes = loaddata(data_dir, batch_size, "validation", shuffle=False)
        with torch.no_grad():
            for inputs, labels in val_loaders["validation"]:
                labels = torch.squeeze(labels.type(torch.LongTensor))
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Forward pass
                outputs = model_ft(inputs)
                loss_val = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)

                # Accumulate loss and predictions
                val_loss += loss_val.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        epoch_val_loss = val_loss / val_sizes
        epoch_val_acc = accuracy_score(all_labels, all_preds)
        epoch_val_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        epoch_val_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        epoch_val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Evaluation of early stopping in validation phase
        early_stopper.step(epoch_val_acc)

        if early_stopper.early_stop:
            print(f"Training stopped early. Best {monitor_metric}: {early_stopper.best}")
            best_model_wts = model_ft.state_dict()
            break

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Output epoch metrics
        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        print(f"Val   Loss: {epoch_val_loss:.4f} | "
              f"Acc: {epoch_val_acc:.4f} | "
              f"Prec: {epoch_val_prec:.4f} | "
              f"Rec: {epoch_val_rec:.4f} | F1: {epoch_val_f1:.4f} | LR: {current_lr:.6f}\n")

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", epoch_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch + 1)
        writer.add_scalar("Loss/val", epoch_val_loss, epoch + 1)
        writer.add_scalar("Accuracy/val", epoch_val_acc, epoch + 1)
        writer.add_scalar("Precision/val", epoch_val_prec, epoch + 1)
        writer.add_scalar("Recall/val", epoch_val_rec, epoch + 1)
        writer.add_scalar("F1/val", epoch_val_f1, epoch + 1)
        writer.add_scalar("LearningRate", current_lr, epoch + 1)
        writer.flush()

        # Build structured log entries for export
        train_logs.append({
            "epoch": epoch + 1,
            "loss": round(epoch_loss, 4),
            "accuracy": round(epoch_acc.item(), 4),
            "learning_rate": current_lr
        })
        val_logs.append({
            "epoch": epoch + 1,
            "val_loss": epoch_val_loss,
            "val_accuracy": epoch_val_acc,
            "val_precision": epoch_val_prec,
            "val_recall": epoch_val_rec,
            "val_f1": epoch_val_f1
        })

        # Save best model if validation improves
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = model_ft.state_dict()

    # Save logs to JSON
    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)
    with open(os.path.join(os.getcwd(), f"logs/train_logs_{timestamp}.json"), "w") as f:
        json.dump(train_logs, f, indent=4)
    with open(os.path.join(os.getcwd(), f"logs/val_logs_{timestamp}.json"), "w") as f:
        json.dump(val_logs, f, indent=4)
    
    # End of training wrapping up
    writer.close()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # Save the best model weights
    os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)
    model_ft.load_state_dict(best_model_wts)
    model_out_path = os.path.join(os.getcwd(), f"models/{net_name}_best_{timestamp}.pth")
    torch.save(model_ft, model_out_path)
    print(f"Model saved to {model_out_path}")

    return train_logs, val_logs, best_model_wts


def test_model(model, criterion):
    """Evaluates the trained model on the test set and saves final metrics."""
    model.eval()
    dset_loaders, dset_sizes = loaddata(data_dir, batch_size, "test", shuffle=False)
    all_preds, all_labels = [], []
    running_loss = 0.0

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

    # Compute metrics
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    test_loss = running_loss / dset_sizes
    test_acc = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    test_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Print model report like first version
    print("\nMODEL REPORT")
    print("-" * 40)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Accuracy:      {test_acc:.4f}")
    print(f"Precision:     {test_prec:.4f}")
    print(f"Recall:        {test_rec:.4f}")
    print(f"F1-Score:      {test_f1:.4f}")
    print("-" * 40)
    print("Training Configuration:")
    print(f" Epochs:       {num_epochs}")
    print(f" Batch Size:   {batch_size}")
    print(f" Learning Rate:{learning_rate}")
    print(f" Momentum:     {momentum}")
    print(f" Weight Decay: {weight_decay}")
    print(f" Gamma:        {gamma}")
    print(f" Model:        {net_name}")
    print(f" Input Size:   {input_size}")
    print("-" * 40)

    # Save report
    report = {
        "timestamp": timestamp,
        "model_name": net_name,
        "input_size": input_size,
        "training_parameters": {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "gamma": gamma,
            "optimizer": "SGD",
            "scheduler": "ExponentialLR"
        },
        "test_metrics": {
            "loss": test_loss,
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
            "f1_score": test_f1
        }
    }

    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)
    with open(os.path.join(os.getcwd(), f"logs/model_report_{timestamp}.json"), "w") as f:
        json.dump(report, f, indent=4)
    print(f"Model report saved as model_report_{timestamp}.json")


def run():
    """Initializes model, optimizer, scheduler, and runs training and testing."""
    if weights_loc:
        model_ft = torch.load(weights_loc)
    else:
        model_ft = EfficientNet.from_pretrained(net_name)

    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, class_num)

    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss().cuda() if use_gpu else nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # Exponential learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # Cosine learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=0)

    # Run training and validation
    train_logs, val_logs, best_model_wts = train_model(model_ft, criterion, optimizer, scheduler, num_epochs=num_epochs)

    print("-" * 10)
    print("Testing best model on validation set...")
    model_ft.load_state_dict(best_model_wts)
    test_model(model_ft, criterion)


if __name__ == "__main__":
    print(f"Dataset: {data_dir} | Epochs: {num_epochs} | Batch size: {batch_size} | Classes: {class_num}")
    run()
