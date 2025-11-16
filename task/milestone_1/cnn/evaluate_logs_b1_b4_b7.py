import json
import os
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_runs(log_dir):
    """Finds all runs in the given directory."""
    runs = {}
    if not os.path.isdir(log_dir):
        return runs
    for name in sorted(os.listdir(log_dir)):
        if not name:
            continue
        if name.startswith('train_logs_') and name.endswith('.json'):
            key = name[len('train_logs_'):-len('.json')]
            runs.setdefault(key, {})['train'] = os.path.join(log_dir, name)
        elif name.startswith('val_logs_') and name.endswith('.json'):
            key = name[len('val_logs_'):-len('.json')]
            runs.setdefault(key, {})['val'] = os.path.join(log_dir, name)
        elif name.startswith('model_report_') and name.endswith('.json'):
            key = name[len('model_report_'):-len('.json')]
            runs.setdefault(key, {})['report'] = os.path.join(log_dir, name)
    return runs

def format_model_name(model_name):
    """Formats the model name for display."""
    if not model_name:
        return ''
    mn = model_name.strip()
    if mn.lower().startswith('efficientnet'):
        parts = mn.split('-', 1)
        if len(parts) == 2:
            return f"EfficientNet {parts[1].upper()}"
        return "EfficientNet"
    return mn.replace('_', ' ').title()

def extract_metrics(train_json, val_json):
    epochs = []
    if train_json:
        for e in train_json:
            epochs.append(e.get('epoch'))
    val_loss = []
    val_acc = []
    if val_json:
        for e in val_json:
            val_loss.append(e.get('val_loss'))
            val_acc.append(e.get('val_accuracy'))
    return epochs, val_loss, val_acc

def select_ordered_keys(runs):
    """Creates a list of keys in order of preference for plotting."""
    desired = ['efficientnet-b1', 'efficientnet-b4', 'efficientnet-b7']
    present = [k for k in desired if k in runs]
    if present:
        return present
    alt_map = {'b1': None, 'b4': None, 'b7': None}
    for k in runs:
        kl = k.lower()
        for tag in alt_map:
            if tag in kl:
                alt_map[tag] = k
    return [alt_map[t] for t in ['b1', 'b4', 'b7'] if alt_map[t]]

def plot_validation_side_by_side(runs, out_dir):
    """Plots the validation loss and accuracy for each model side-by-side."""
    keys = select_ordered_keys(runs)
    if not keys:
        print("No matching runs for B1/B4/B7")
        return

    series = []
    for k in keys:
        paths = runs[k]
        train_p = paths.get('train')
        val_p = paths.get('val')
        rep_p = paths.get('report')
        train_json = load_json(train_p) if train_p and os.path.exists(train_p) else None
        val_json = load_json(val_p) if val_p and os.path.exists(val_p) else None
        report = load_json(rep_p) if rep_p and os.path.exists(rep_p) else None
        epochs, v_loss, v_acc = extract_metrics(train_json, val_json)
        if not v_loss and not v_acc:
            print(f"Skipping {k}: no validation data")
            continue
        label = k
        if report and isinstance(report, dict):
            model = report.get('model_name')
            if model:
                label = format_model_name(model)

        x = epochs if epochs and len(epochs) >= max(len(v_loss), len(v_acc)) else list(range(1, max(len(v_loss), len(v_acc)) + 1))
        series.append((x[:len(v_loss)], v_loss, x[:len(v_acc)], v_acc, label))

    if not series:
        print("No validation data to plot")
        return

    os.makedirs(out_dir, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    plot_loss(ax_loss, series)
    plot_accuracy(ax_acc, series)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "efficientnet_b1_b4_b7_val_loss_accuracy.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved combined plot: {out_path}")


def plot_accuracy(ax_acc, series: list):
    """Plots the accuracy for each model"""
    for i, (_, _, x_acc, y_acc, label) in enumerate(series):
        ax_acc.plot(x_acc, y_acc, linewidth=2, label=label, color=f"C{i}")
    ax_acc.set_title("Validation Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("val accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend()


def plot_loss(ax_loss, series: list):
    """Plots the loss for each model"""
    for i, (x_loss, y_loss, _, _, label) in enumerate(series):
        ax_loss.plot(x_loss, y_loss, linewidth=2, label=label, color=f"C{i}")
    ax_loss.set_title("Validation Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("val loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()


def main():
    logs_directory = 'logs'
    output_directory = 'logs/plots'
    runs = find_runs(logs_directory)
    plot_validation_side_by_side(runs, output_directory)

if __name__ == '__main__':
    main()
