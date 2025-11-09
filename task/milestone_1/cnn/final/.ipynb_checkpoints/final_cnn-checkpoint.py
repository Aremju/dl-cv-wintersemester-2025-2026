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


# ----------------------------- CONFIG -----------------------------
use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === Hyperparameters ===
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
gamma = 0.9  # exponential decay factor

# Timestamp for saving logs uniquely
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ----------------------------- DATA LOADING -----------------------------
def loaddata(data_dir, batch_size, set_name, shuffle):
    if set_name.lower() == "train":
        folder = os.path.join(data_dir, "train")
    elif set_name.lower() == "validation":
        folder = os.path.join(data_dir, "validation")
    elif set_name.lower() == "test":
        folder = os.path.join(data_dir, "test")
    else:
        raise ValueError(f"Unknown set_name: {set_name}")

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

    dataset = datasets.ImageFolder(folder, transform=data_transforms[set_name.lower()])

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    dataset_size = len(dataset)
    print(f"Loaded '{set_name}' from: {folder} ({dataset_size} images, {len(dataset.classes)} classes)")
    return {set_name: dataloader}, dataset_size


# ----------------------------- TRAINING FUNCTION -----------------------------
def train_model(model_ft, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    train_logs, val_logs = [], []

    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), "runs"))

    for epoch in range(epoch_to_resume_from, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # --- TRAINING PHASE ---
        model_ft.train()
        dset_loaders, dset_sizes = loaddata(data_dir, batch_size, "train", shuffle=True)

        running_loss = 0.0
        running_corrects = 0
        count = 0

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

            if count % batch_size == 0:
                batch_acc = (running_corrects.double() / (count * batch_size)).item()
                print(f"[Batch {count}] Loss: {loss.item():.4f} | Train Acc: {batch_acc:.4f}")

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes

        # --- VALIDATION PHASE ---
        model_ft.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        val_loaders, val_sizes = loaddata(data_dir, batch_size, "validation", shuffle=False)
        with torch.no_grad():
            for inputs, labels in val_loaders["validation"]:
                labels = torch.squeeze(labels.type(torch.LongTensor))
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model_ft(inputs)
                loss_val = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)

                val_loss += loss_val.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        epoch_val_loss = val_loss / val_sizes
        epoch_val_acc = accuracy_score(all_labels, all_preds)
        epoch_val_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        epoch_val_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        epoch_val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- Logging ---
        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        print(f"Val   Loss: {epoch_val_loss:.4f} | "
              f"Acc: {epoch_val_acc:.4f} | "
              f"Prec: {epoch_val_prec:.4f} | "
              f"Rec: {epoch_val_rec:.4f} | F1: {epoch_val_f1:.4f} | LR: {current_lr:.6f}\n")

        # --- TensorBoard ---
        #writer.add_scalar("Loss/train", epoch_loss, epoch)
        #writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        #writer.add_scalar("Loss/val", epoch_val_loss, epoch)
        #writer.add_scalar("Accuracy/val", epoch_val_acc, epoch)
        #writer.add_scalar("Precision/val", epoch_val_prec, epoch)
        #writer.add_scalar("Recall/val", epoch_val_rec, epoch)
        #writer.add_scalar("F1/val", epoch_val_f1, epoch)
        #writer.add_scalar("LearningRate", current_lr, epoch)

        # --- Save logs for JSON ---
        train_logs.append({
            "epoch": f"{epoch + 1}",
            "loss": f"{epoch_loss:.4f}",
            "accuracy": f"{epoch_acc:.4f}",
            "learning_rate": f"{current_lr}",
        })
        val_logs.append({
            "epoch": epoch + 1,
            "val_loss": epoch_val_loss,
            "val_accuracy": epoch_val_acc,
            "val_precision": epoch_val_prec,
            "val_recall": epoch_val_rec,
            "val_f1": epoch_val_f1
        })

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = model_ft.state_dict()

    # --- Save logs as JSON ---
    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)
    with open(os.path.join(os.getcwd(), f"logs/train_logs_{timestamp}.json"), "w") as f:
        json.dump(train_logs, f, indent=4)
    with open(os.path.join(os.getcwd(), f"logs/val_logs_{timestamp}.json"), "w") as f:
        json.dump(val_logs, f, indent=4)

    # --- Wrap up ---
    writer.close()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model_ft.load_state_dict(best_model_wts)
    model_out_path = os.path.join(os.getcwd(), f"{net_name}_best_{timestamp}.pth")
    torch.save(model_ft, model_out_path)
    print(f"Model saved to {model_out_path}")

    return train_logs, val_logs, best_model_wts


# ----------------------------- TEST FUNCTION -----------------------------
def test_model(model, criterion):
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

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    # --- Final test metrics ---
    test_loss = running_loss / dset_sizes
    test_acc = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    test_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

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

    # Save to JSON (metrics + parameters)
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
    with open(os.path.join(os.getcwd(), f"logs/model_report_{timestamp}.json"), "w") as f:
        json.dump(report, f, indent=4)
    print(f"Model report saved as model_report_{timestamp}.json")


# ----------------------------- MAIN TRAINING WRAPPER -----------------------------
def run():
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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    train_logs, val_logs, best_model_wts = train_model(model_ft, criterion, optimizer, scheduler, num_epochs=num_epochs)

    print("-" * 10)
    print("Testing best model on validation set...")
    model_ft.load_state_dict(best_model_wts)
    test_model(model_ft, criterion)


# ----------------------------- RUN -----------------------------
if __name__ == "__main__":
    print(f"Dataset: {data_dir} | Epochs: {num_epochs} | Batch size: {batch_size} | Classes: {class_num}")
    run()
