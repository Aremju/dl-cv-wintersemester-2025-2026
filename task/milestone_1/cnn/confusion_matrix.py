import time
import os
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MODEL_PATH = "efficientnet-b7_best_2025-11-17_05-08-07.pth"
TEST_DATASET_PATH = "../../../data/animal_images/test"
IMAGE_SIZE = 380
BATCH_SIZE = 1
NORMALIZE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CM_SAVE_PATH = "confusion_matrix.npy"


def compute_confusion_matrix(model, test_loader):
    """Runs inference and computes confusion matrix."""
    y_true = []
    y_pred = []

    total_batches = len(test_loader)
    start_time = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            elapsed = time.time() - start_time
            batches_done = i + 1
            batches_left = total_batches - batches_done
            avg_time_per_batch = elapsed / batches_done
            eta_seconds = avg_time_per_batch * batches_left

            print(
                f"Batch {batches_done}/{total_batches} - "
                f"ETA: {eta_seconds:.1f}s "
                f"(avg {avg_time_per_batch:.3f}s/batch)"
            )

    return confusion_matrix(y_true, y_pred, normalize=NORMALIZE)


def plot_confusion_matrix_torch():
    """Plots the confusion matrix for a torch model using scikit-learn."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    test_dataset = datasets.ImageFolder(TEST_DATASET_PATH, transform=transform)
    class_names = test_dataset.classes

    if os.path.exists(CM_SAVE_PATH):
        print(f"Loading saved confusion matrix: {CM_SAVE_PATH}")
        cm = np.load(CM_SAVE_PATH)
    else:
        print("No saved confusion matrix found — running evaluation.")
        model = torch.load(MODEL_PATH, map_location=DEVICE)
        model.eval()
        model.to(DEVICE)

        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        cm = compute_confusion_matrix(model, test_loader)

        # save confusion matrix to prevent re-computation
        np.save(CM_SAVE_PATH, cm)
        print(f"Confusion matrix saved to {CM_SAVE_PATH}")

    # plotting confusion matrix like in vit
    fig, ax = plt.subplots(figsize=(10, 9))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        cmap="Blues",
        xticks_rotation=70,
        values_format=".2f" if NORMALIZE else "d",
        ax=ax,
    )

    ax.set_xlabel("Predicted Labels", fontsize=15, labelpad=15)
    ax.set_ylabel("True Labels", fontsize=15, labelpad=15)
    ax.set_title("Confusion Matrix (Animal_Images)", fontsize=16, pad=10)

    ax.tick_params(axis="x", labelsize=11, pad=5)
    ax.tick_params(axis="y", labelsize=11, pad=5)

    plt.tight_layout()

    output_path = "confusion_matrix_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    plt.show()

if __name__ == "__main__":
    plot_confusion_matrix_torch()
