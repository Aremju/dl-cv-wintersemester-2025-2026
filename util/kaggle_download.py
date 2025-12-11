import os
import shutil
import kagglehub

# Target directory
target_dir = "../../data/imagenet"

# Temporary download of dataset
print("Downloading dataset")
path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")

print(f"Downloaded dataset to: {path}")

os.makedirs(target_dir, exist_ok=True)

# Copy of data from temporary into target directory
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(target_dir, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print(f"Successful copy in '{target_dir}'!")
print(f"Path: {os.path.abspath(target_dir)}")
