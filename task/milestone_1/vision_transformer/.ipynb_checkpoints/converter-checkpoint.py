import os
from PIL import Image

def convert_to_png_recursive(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            name, ext = os.path.splitext(file)
            ext_lower = ext.lower()
            if ext_lower in ('.jpg', '.jpeg'):
                file_path = os.path.join(dirpath, file)
                new_file_name = f"{name}_{ext_lower[1:]}.png"
                new_file_path = os.path.join(dirpath, new_file_name)

                try:
                    # Open and convert
                    img = Image.open(file_path).convert("RGBA")
                    img.save(new_file_path, "PNG")

                    # Verify PNG exists before deletion
                    if os.path.exists(new_file_path):
                        os.remove(file_path)
                        print(f"✅ Converted & removed: {file_path} → {new_file_path}")
                    else:
                        print(f"⚠️ PNG not found after saving: {new_file_path}")

                except Exception as e:
                    print(f"❌ Failed to convert {file_path}: {e}")

if __name__ == "__main__":
    folder = "../../../data/animal_images/Training Data/train"
    if os.path.exists(folder):
        convert_to_png_recursive(folder)
        print("🎉 Conversion completed and originals removed.")
    else:
        print("❌ Folder not found:", folder)
