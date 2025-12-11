import os
import re
import json
import matplotlib.pyplot as plt

LOG_DIR = os.path.join('logs_lr')
OUT_DIR = os.path.join(LOG_DIR, 'plots')


def load_json(path):
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_scheduler_runs(log_dir):
    """Group files by scheduler key using multiple naming patterns."""
    runs = {}
    if not os.path.isdir(log_dir):
        return runs

    patterns = [
        re.compile(r'^(?P<key>.+)_train_logs_.*\.json$'),
        re.compile(r'^(?P<key>.+)_val_logs_.*\.json$'),
        re.compile(r'^(?P<key>.+)_model_report_.*\.json$'),
        re.compile(r'^train_logs_(?P<key>.+)\.json$'),
        re.compile(r'^val_logs_(?P<key>.+)\.json$'),
        re.compile(r'^model_report_(?P<key>.+)\.json$'),
    ]

    for name in sorted(os.listdir(log_dir)):
        if not name.endswith('.json'):
            continue

        matched = False
        for p in patterns:
            m = p.match(name)
            if not m:
                continue

            key = m.group('key')

            if '_val_logs_' in name or name.startswith('val_logs_'):
                runs.setdefault(key, {})['val'] = os.path.join(log_dir, name)
            elif '_model_report_' in name or name.startswith('model_report_'):
                runs.setdefault(key, {})['report'] = os.path.join(log_dir, name)
            elif '_train_logs_' in name or name.startswith('train_logs_'):
                runs.setdefault(key, {})['train'] = os.path.join(log_dir, name)

            matched = True
            break

        if not matched:
            if '_val_logs_' in name:
                key = name.split('_val_logs_')[0]
                runs.setdefault(key, {})['val'] = os.path.join(log_dir, name)
            elif '_model_report_' in name:
                key = name.split('_model_report_')[0]
                runs.setdefault(key, {})['report'] = os.path.join(log_dir, name)
            elif '_train_logs_' in name:
                key = name.split('_train_logs_')[0]
                runs.setdefault(key, {})['train'] = os.path.join(log_dir, name)

    return runs


def pretty_label(key):
    """Map internal keys to pretty scheduler names."""
    mapping = {
        'cos': 'Cosine LR',
        'cosine': 'Cosine LR',
        'exp': 'Exponential LR',
        'exponential': 'Exponential LR',
        'static': 'Static LR',
        'static_lr': 'Static LR'
    }
    return mapping.get(key.lower(), key)


def extract_val_metrics(val_path):
    """Extract epochs, validation loss and validation accuracy."""
    if not val_path or not os.path.exists(val_path):
        return [], [], []

    data = load_json(val_path)
    epochs, loss, acc = [], [], []

    for e in data:
        epochs.append(e.get("epoch"))
        loss.append(e.get("val_loss"))
        acc.append(e.get("val_accuracy"))

    return epochs, loss, acc


def plot_lr_scheduler_val_metrics(log_dir, out_dir):
    """Plot validation accuracy and loss for all detected schedulers."""
    runs = find_scheduler_runs(log_dir)
    if not runs:
        return

    os.makedirs(out_dir, exist_ok=True)

    fig, (ax_acc, ax_loss) = plt.subplots(
        1, 2, figsize=(9, 4), sharey=False
    )

    for i, (key, paths) in enumerate(runs.items()):
        val_path = paths.get("val")
        if not val_path:
            continue

        epochs, val_loss, val_acc = extract_val_metrics(val_path)
        label = pretty_label(key)

        ax_acc.plot(
            epochs,
            val_acc,
            linewidth=2,
            label=label,
            color=f"C{i}"
        )

        ax_loss.plot(
            epochs,
            val_loss,
            linewidth=2,
            label=label,
            color=f"C{i}"
        )

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Val Accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend()

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Val Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    fig.tight_layout()

    out_path = os.path.join(out_dir, "scheduler_val_metrics.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    plot_lr_scheduler_val_metrics(LOG_DIR, OUT_DIR)
