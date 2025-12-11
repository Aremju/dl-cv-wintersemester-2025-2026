from PIL import Image

cat_path = "./paper_images/multi_cats.png"
dog_path = "./paper_images/multi_dogs.png"

cat = Image.open(cat_path)
dog = Image.open(dog_path)

images = [cat, dog]

w, h = images[0].size
images_resized = [img.resize((w, h)) for img in images]

combined = Image.new("RGB", (w * 2, h * 1))

combined.paste(images_resized[0], (0, 0))
combined.paste(images_resized[1], (w, 0))

output_path = "cats_dogs_combined.png"
combined.save(output_path)

print("Saved combined image as:", output_path)
