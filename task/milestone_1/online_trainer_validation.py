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


# ----------------------------- CONFIG -----------------------------

use_gpu = torch.cuda.is_available()
print("Using GPU:", use_gpu)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Default hyperparameters
data_dir = ""
num_epochs = 40
batch_size = 4
input_size = (380, 380)  # EfficientNet-B4 default input size
class_num = 15
weights_loc = None
lr = 0.001
net_name = "efficientnet-b4"
epoch_to_resume_from = 0
momentum = 0.9


# ----------------------------- DATA LOADING -----------------------------

def loaddata(data_dir, batch_size, set_name, shuffle):
    """
    Loads images from the nested folder structure:
    data/animal_images/Training Data/Training Data/...classes...
    data/animal_images/Validation Data/Validation Data/...classes...
    """

    if set_name.lower() in ["train", "training", "training data"]:
        folder = os.path.join(data_dir, "Training Data", "Training Data")
    elif set_name.lower() in ["val", "validation", "test", "validation data"]:
        folder = os.path.join(data_dir, "Validation Data", "Validation Data")
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
        "test": transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    dataset = datasets.ImageFolder(
        folder,
        transform=data_transforms["train" if "train" in set_name.lower() else "test"]
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 0 avoids shared memory issues in Kaggle/Docker
    )

    dataset_size = len(dataset)
    print(f"Loaded '{set_name}' from: {folder} ({dataset_size} images, {len(dataset.classes)} classes)")
    return {set_name: dataloader}, dataset_size


# ----------------------------- TRAINING FUNCTION -----------------------------

def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50):
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    train_loss_list = []

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(data_dir, "runs"))

    for epoch in range(epoch_to_resume_from, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # --- Training phase ---
        model_ft.train(True)
        dset_loaders, dset_sizes = loaddata(data_dir, batch_size, "train", shuffle=True)
        optimizer = lr_scheduler(optimizer, epoch)

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
                print(f"[Batch {count}] Loss: {loss.item():.4f}")

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes

        # --- Validation phase ---
        model_ft.eval()
        val_loss = 0.0
        val_corrects = 0

        val_loaders, val_sizes = loaddata(data_dir, batch_size, "val", shuffle=False)
        with torch.no_grad():
            for inputs, labels in val_loaders["val"]:
                labels = torch.squeeze(labels.type(torch.LongTensor))
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model_ft(inputs)
                loss_val = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)

                val_loss += loss_val.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / val_sizes
        epoch_val_acc = val_corrects.double() / val_sizes

        # --- Logging ---
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(f"Val   Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}\n")

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        writer.add_scalar("Loss/val", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/val", epoch_val_acc, epoch)

        train_loss_list.append(epoch_loss)

        # --- Save best model ---
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = model_ft.state_dict()

    writer.close()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # Save final best model
    os.makedirs(os.path.join(data_dir, "model"), exist_ok=True)
    model_ft.load_state_dict(best_model_wts)
    model_out_path = os.path.join(data_dir, "model", f"{net_name}.pth")
    torch.save(model_ft, model_out_path)
    print(f"✅ Model saved to {model_out_path}")

    return train_loss_list, best_model_wts


# ----------------------------- TEST FUNCTION -----------------------------

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

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    print(f"Test Loss: {running_loss / dset_sizes:.4f} Acc: {running_corrects.double() / dset_sizes:.4f}")


# ----------------------------- LR SCHEDULER -----------------------------

def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decays learning rate by 0.8 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8 ** (epoch // lr_decay_epoch))
    print(f"LR is set to {lr}")
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


# ----------------------------- MAIN TRAINING WRAPPER -----------------------------

def run():
    if weights_loc:
        model_ft = torch.load(weights_loc)
    else:
        model_ft = EfficientNet.from_pretrained(net_name)

    # Modify final FC layer
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, class_num)

    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss().cuda() if use_gpu else nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum, weight_decay=0.0004)

    train_loss, best_model_wts = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

    print("-" * 10)
    print("Testing best model on validation set...")
    model_ft.load_state_dict(best_model_wts)
    test_model(model_ft, criterion)


# ----------------------------- ARG PARSER -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../../data/animal_images", help="root folder of dataset")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--class-num", type=int, default=15)
    parser.add_argument("--weights-loc", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--net-name", type=str, default="efficientnet-b4")
    parser.add_argument("--resume-epoch", type=int, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)

    opt = parser.parse_args()

    data_dir = opt.data_dir
    num_epochs = opt.num_epochs
    batch_size = opt.batch_size
    class_num = opt.class_num
    weights_loc = opt.weights_loc
    lr = opt.lr
    net_name = opt.net_name
    epoch_to_resume_from = opt.resume_epoch
    momentum = opt.momentum

    print(f"Dataset: {data_dir} | Epochs: {num_epochs} | Batch size: {batch_size} | Classes: {class_num}")
    run()
