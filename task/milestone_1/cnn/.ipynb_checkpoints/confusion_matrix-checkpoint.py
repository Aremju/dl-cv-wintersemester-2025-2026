import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# adjust settings here
MODEL_PATH = "./models/efficientnet-b7_best_2025-11-17_05-08-07.pth"
TEST_DATASET_PATH = "../../../data/animal_images/test"
IMAGE_SIZE = 380
BATCH_SIZE = 32
NORMALIZE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_confusion_matrix_torch():
    """Plots the confusion matrix for a torch model using scikit-learn."""
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.eval()
    model.to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(TEST_DATASET_PATH, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = test_dataset.classes

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize=NORMALIZE)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format=".2f" if NORMALIZE else "d")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_confusion_matrix_torch()
