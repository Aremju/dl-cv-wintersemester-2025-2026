from PIL import Image

cat1_path = "./paper_images/cats1.jpeg"
cat2_path = "./paper_images/cats2.jpeg"
mouse1_path = "./paper_images/mice1.png"
mouse2_path = "./paper_images/mice2.png"

cat1 = Image.open(cat1_path)
mouse1 = Image.open(mouse1_path)
mouse2 = Image.open(mouse2_path)
cat2 = Image.open(cat2_path)

images = [cat1, mouse1, mouse2, cat2]

w, h = images[0].size
images_resized = [img.resize((w, h)) for img in images]

combined = Image.new("RGB", (w * 2, h * 2))

combined.paste(images_resized[0], (0, 0))
combined.paste(images_resized[1], (w, 0))
combined.paste(images_resized[2], (0, h))
combined.paste(images_resized[3], (w, h))

output_path = "cats_mice_combined.png"
combined.save(output_path)

print("Saved combined image as:", output_path)
