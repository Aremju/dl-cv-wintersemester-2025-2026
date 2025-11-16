# python
import json
import os
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_runs(log_dir):
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
    train_loss = []
    train_acc = []
    if train_json:
        for e in train_json:
            epochs.append(e.get('epoch'))
            train_loss.append(e.get('loss'))
            train_acc.append(e.get('accuracy'))
    val_loss = []
    val_acc = []
    if val_json:
        for e in val_json:
            val_loss.append(e.get('val_loss'))
            val_acc.append(e.get('val_accuracy'))
    return epochs, train_loss, train_acc, val_loss, val_acc

def plot_combined(runs, out_dir):
    keys = sorted(runs.keys())
    if not keys:
        print("No runs to plot")
        return

    n = len(keys)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 4 * n), squeeze=False)

    titles = [''] * n

    for i, key in enumerate(keys):
        paths = runs[key]
        train_p = paths.get('train')
        val_p = paths.get('val')
        rep_p = paths.get('report')

        train_json = load_json(train_p) if train_p and os.path.exists(train_p) else None
        val_json = load_json(val_p) if val_p and os.path.exists(val_p) else None
        report = load_json(rep_p) if rep_p and os.path.exists(rep_p) else None

        epochs, t_loss, t_acc, v_loss, v_acc = extract_metrics(train_json, val_json)
        if not epochs:
            print(f"Skipping {key}: no training data")
            continue

        ax_loss = axes[i][0]
        ax_acc = axes[i][1]

        ax_loss.plot(epochs, t_loss, '-', color='C0', linewidth=2, label='train loss')
        if v_loss:
            ax_loss.plot(epochs[:len(v_loss)], v_loss, '-', color='C1', linewidth=2, label='val loss')
        ax_loss.set_ylabel('loss')
        ax_loss.grid(alpha=0.3)
        ax_loss.legend(loc='upper right')

        ax_acc.plot(epochs, t_acc, '-', color='C2', linewidth=2, label='train acc')
        if v_acc:
            ax_acc.plot(epochs[:len(v_acc)], v_acc, '-', color='C3', linewidth=2, label='val acc')
        ax_acc.set_ylabel('accuracy')
        ax_acc.grid(alpha=0.3)
        ax_acc.legend(loc='lower right')

        title = key
        if report and isinstance(report, dict):
            model = report.get('model_name')
            if model:
                title = format_model_name(model)

        titles[i] = title

        ax_acc.set_xlabel('epoch')
        ax_loss.set_xlabel('epoch')

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for i, t in enumerate(titles):
        if not t:
            continue
        ax_l = axes[i][0]
        ax_r = axes[i][1]
        pos_l = ax_l.get_position()
        pos_r = ax_r.get_position()
        x_center = (pos_l.x0 + pos_r.x1) / 2.0
        y = pos_l.y1 + 0.01
        fig.text(x_center, y, t, ha='center', va='bottom', fontsize=12)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'efficientnet_b1_b4_b7_combined.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved combined plot: {out_path}")

def main():
    logs = 'logs'
    out = 'logs/plots'
    runs = find_runs(logs)
    plot_combined(runs, out)

if __name__ == '__main__':
    main()
