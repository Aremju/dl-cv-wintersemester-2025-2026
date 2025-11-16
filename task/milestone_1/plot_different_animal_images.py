import matplotlib.pyplot as plt
import os

path_to_tiger_train_image = "../../data/animal_images/train/Tiger/Tiger (361).jpeg"
path_to_elephant_train_image = "../../data/animal_images/train/Elephant/Elephant-Train (43).jpeg"
path_to_lizard_train_image = "../../data/animal_images/train/Lizard/Lizard-Train (1).png"
path_to_cow_train_image = "../../data/animal_images/train/Cow/Cow-Train (5).jpeg"
path_to_hippo_train_image = "../../data/animal_images/train/Hippo/Hippo - Train (36).jpeg"
path_to_beetle_train_image = "../../data/animal_images/train/Beetle/Beetle-Train (8).jpg"

paths_with_label = [
    (path_to_tiger_train_image, "Tiger"),
    (path_to_elephant_train_image, "Elephant"),
    (path_to_lizard_train_image, "Lizard"),
    (path_to_cow_train_image, "Cow"),
    (path_to_hippo_train_image, "Hippo"),
    (path_to_beetle_train_image, "Beetle")
]

def plot_images_with_labels_as_grid(paths_and_labels):
    """Plots images with labels as a grid."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.ravel()

    for i, (path, label) in enumerate(paths_and_labels):
        if not os.path.exists(path):
            axes[i].text(0.5, 0.5, "File not found", ha="center", va="center")
            axes[i].set_title(label)
            axes[i].axis("off")
            continue

        image = plt.imread(path)
        axes[i].imshow(image)
        axes[i].set_title(label)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("different_animal_images.png", bbox_inches="tight", pad_inches=0)
    plt.show()

if __name__ == "__main__":
    plot_images_with_labels_as_grid(paths_with_label)
