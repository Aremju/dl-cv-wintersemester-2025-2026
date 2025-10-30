from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
import os
import argparse
from efficientnet_pytorch import EfficientNet  # pip install efficientnet-pytorch

# ============================================================
# Basic Settings
# ============================================================
use_gpu = torch.cuda.is_available()
print("GPU available:", use_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Default parameters
data_dir = "../../data/animal_images"
num_epochs = 25
batch_size = 8
input_size = 380
class_num = 15
weights_loc = None
lr = 0.001
net_name = 'efficientnet-b4'
epoch_to_resume_from = 0
momentum = 0.9

# ============================================================
# Data Loader
# ============================================================
def loaddata(data_dir, batch_size, set_name, shuffle=True):
    """Load train, validation or test datasets."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # Build correct path depending on dataset split
    if set_name.lower().startswith("train"):
        data_path = os.path.join(data_dir, "Training Data", "Training Data")
    elif set_name.lower().startswith("val"):
        data_path = os.path.join(data_dir, "Validation Data", "Validation Data")
    elif set_name.lower().startswith("test"):
        data_path = os.path.join(data_dir, "Testing Data", "Testing Data")
    else:
        raise ValueError(f"Unknown set_name: {set_name}")

    transform_key = 'train' if set_name.lower().startswith("train") else (
        'val' if set_name.lower().startswith("val") else 'test')
    dataset = datasets.ImageFolder(data_path, data_transforms[transform_key])
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=2, pin_memory=True
    )
    return loader, len(dataset)

# ============================================================
# Training
# ============================================================
def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50):
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0

    for epoch in range(epoch_to_resume_from, num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        optimizer = lr_scheduler(optimizer, epoch)

        # --- Training ---
        model_ft.train()
        train_loader, train_size = loaddata(data_dir, batch_size, 'Training', shuffle=True)
        val_loader, val_size = loaddata(data_dir, batch_size, 'Validation', shuffle=False)

        running_loss, running_corrects = 0.0, 0
        count = 0
        for inputs, labels in train_loader:
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.data, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            count += 1
            if count % 30 == 0:
                
                print(f'Batch Nr. [{count}] Loss: {loss}')
            
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')

        # --- Validation ---
        model_ft.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model_ft(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= val_size
        val_acc = val_corrects.double() / val_size
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        # Save the best model weights
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model_ft.state_dict()
            torch.save(best_model_wts, os.path.join(data_dir, 'best_efficientnet_b4.pth'))
            print("✓ Best model saved!")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    model_ft.load_state_dict(best_model_wts)
    return model_ft

# ============================================================
# Testing
# ============================================================
def test_model(model, criterion):
    """Evaluate model performance on the separate test set."""
    print("\n--- Testing on 'Testing Data' ---")
    model.eval()
    test_loader, test_size = loaddata(data_dir, batch_size, 'Testing', shuffle=False)
    running_loss, running_corrects = 0.0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / test_size
    test_acc = running_corrects.double() / test_size
    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')

# ============================================================
# Learning Rate Scheduler
# ============================================================
def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=10):
    """Decay learning rate by factor 0.8 every 'lr_decay_epoch' epochs."""
    lr = init_lr * (0.8 ** (epoch // lr_decay_epoch))
    print(f'Learning Rate set to {lr:.6f}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# ============================================================
# Main Execution
# ============================================================
def run():
    global data_dir, num_epochs, batch_size, input_size, class_num, lr, net_name, weights_loc, momentum

    # Load pretrained model weights or create a new one
    if weights_loc:
        print(f"Loading weights from {weights_loc}")
        model_ft = torch.load(weights_loc)
    else:
        model_ft = EfficientNet.from_pretrained(net_name)
        print(f"Loaded pretrained EfficientNet: {net_name}")

    # Replace fully connected layer with the number of classes
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, class_num)

    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=lr,
                          momentum=momentum, weight_decay=0.0004)

    # Train + Validate
    model_ft = train_model(model_ft, criterion, optimizer,
                           exp_lr_scheduler, num_epochs=num_epochs)

    # Test on separate test dataset
    test_model(model_ft, criterion)

# ============================================================
# CLI Arguments
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="../../data/animal_images")
    parser.add_argument('--num-epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=380)
    parser.add_argument('--class-num', type=int, default=15)
    parser.add_argument('--weights-loc', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--net-name', type=str, default="efficientnet-b4")
    parser.add_argument('--resume-epoch', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    opt = parser.parse_args()

    # Apply CLI arguments
    data_dir = opt.data_dir
    num_epochs = opt.num_epochs
    batch_size = opt.batch_size
    input_size = opt.img_size
    class_num = opt.class_num
    weights_loc = opt.weights_loc
    lr = opt.lr
    net_name = opt.net_name
    epoch_to_resume_from = opt.resume_epoch
    momentum = opt.momentum

    print(f"Training {net_name} with {class_num} classes on {data_dir}")
    run()
