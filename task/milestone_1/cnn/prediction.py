import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import os

from util.model_loading.model_loader import load_model, DEVICE

# ---------------- CONFIG ----------------

# IMAGE_PATH = "../../data/animal_images/Testing Data/Testing Data/Elephant/Elephant-Test (418).jpg"  # Image to classify
IMAGE_PATH = "../kekw.jpg"  # Image to classify
CLASS_NAMES = [
    "Beetle", "Butterfly", "Cat", "Cow", "Dog", "Elephant", "Gorilla",
    "Hippo", "Lizard", "Monkey", "Mouse", "Panda", "Spider", "Tiger", "Zebra"
]
INPUT_SIZE = (380, 380)  # EfficientNet-B4 input size
NET_NAME = "efficientnet-b4"

# ---------------- LOAD MODEL ----------------
print("Loading model...")

# 1️⃣ Recreate the same EfficientNet-B4 architecture used during training
model = EfficientNet.from_name(NET_NAME)
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

# 2️⃣ Load state_dict weights (your .pth file stores only parameters)
state_dict = load_model()

# Handle possible wrappers (e.g., if saved from DataParallel)
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE)
model.eval()

print("✅ Model loaded and ready for inference!")

# ---------------- IMAGE PREPROCESSING ----------------
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """Loads an image, preprocesses it, and returns the predicted class."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx].item()

    print(f"\nPrediction: 🐾 {predicted_class} ({confidence*100:.2f}% confidence)")
    return predicted_class, confidence


# ---------------- RUN EXAMPLE ----------------
if __name__ == "__main__":
    predict_image(IMAGE_PATH)
