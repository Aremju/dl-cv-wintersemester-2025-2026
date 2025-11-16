import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


def find_latest_event_file(tensorboard_log_directory: str) -> str:
    """Finds the latest .tfevents file in a given directory."""
    event_files = [f for f in os.listdir(tensorboard_log_directory) if f.startswith("events.out.tfevents")]
    if not event_files:
        raise FileNotFoundError(f"Keine .tfevents-Datei in '{tensorboard_log_directory}' gefunden.")
    event_files = sorted(event_files, key=lambda f: os.path.getmtime(os.path.join(tensorboard_log_directory, f)))
    return os.path.join(tensorboard_log_directory, event_files[-1])


def load_scalars_from_log(log_path: str) -> dict:
    """Loads scalar data from a TensorBoard log file."""
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    scalar_dict = {}
    for tag in tags:
        data = ea.Scalars(tag)
        scalar_dict[tag] = {
            "steps": [s.step for s in data],
            "values": [s.value for s in data],
        }
    return scalar_dict


def plot_metric(scalar_dict: dict, metric_names: dict, title: str = None, ylabel: str = None):
    """Plots a tensorboard metric as a line plot in matplotlib."""
    plt.figure(figsize=(8, 5))
    for key in metric_names.keys():
        if key not in scalar_dict:
            print(f"Could not find '{key}' in logfiles")
            continue
        plt.plot(scalar_dict[key]["steps"], scalar_dict[key]["values"], label=metric_names[key], linewidth=2)
    plt.xlabel("Step")
    plt.ylabel(ylabel or "Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    log_dir = "output-models/runs/Nov08_19-47-29_gpu-jupyterlab-lunruh-74bf78ccfd-x2fkr"
    log_file = find_latest_event_file(log_dir)
    print(f"Used log file: {log_file}")
    scalars = load_scalars_from_log(log_file)
    print("Available metrics:", list(scalars.keys()))
    # plot needed metrics for the paper
    plot_metric(scalars, {"train/loss": "train", "eval/loss": "val"}, title="Train vs Validation Loss", ylabel="Loss")
    plot_metric(scalars, {"eval/accuracy": "val"}, title="Train vs Validation Accuracy", ylabel="Accuracy")
