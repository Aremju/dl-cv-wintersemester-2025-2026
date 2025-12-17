import os

def rename_images_to_jpg_recursive(base_folder):
    # Changes to following formats
    extensions = [".jpeg", ".png", ".jfif", ".heic", ".webp"]

    for root, _, files in os.walk(base_folder):
        for filename in files:
            old_path = os.path.join(root, filename)
            name, ext = os.path.splitext(filename)
            ext_lower = ext.lower()

            if ext_lower in extensions:
                new_name = f"{name}.jpg"
                new_path = os.path.join(root, new_name)

                # If image already exists put number after
                counter = 1
                while os.path.exists(new_path):
                    new_name = f"{name}_{counter}.jpg"
                    new_path = os.path.join(root, new_name)
                    counter += 1

                os.rename(old_path, new_path)
                print(f"Umbenannt: {old_path} → {new_path}")

rename_images_to_jpg_recursive("../data/Training Data/train")