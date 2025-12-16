import glob
import os

from matplotlib import pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

VAL_ACC = "eval/accuracy"
VAL_LOSS = "eval/loss"

def load_scalars(event_dir, tag):
    """Loads scalar values from a TensorBoard event file."""
    event_files = glob.glob(os.path.join(event_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        raise FileNotFoundError(f"No eventfile found in {event_dir}.")
    accumulator = EventAccumulator(event_files[0])
    accumulator.Reload()
    if tag not in accumulator.Tags()["scalars"]:
        raise ValueError(f"tag '{tag}' not available")
    events = accumulator.Scalars(tag)
    values = [e.value for e in events]
    epochs = list(range(1, len(values) + 1))
    return epochs, values

def plot_comparison(group_dict, title_prefix):
    """Creates two diagrams with validation accuracy and loss for each group
       and saves them as PNG."""
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 2)
    for name, path in group_dict.items():
        epochs, vals = load_scalars(path, VAL_LOSS)
        plt.plot(epochs, vals, label=name)
    plt.title(f"{title_prefix} – Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 1)
    for name, path in group_dict.items():
        epochs, vals = load_scalars(path, VAL_ACC)
        plt.plot(epochs, vals, label=name)
    plt.title(f"{title_prefix} – Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    safe_name = title_prefix.replace(" ", "_")
    filename = f"{safe_name}_vit.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved as: {filename}")

    plt.show()