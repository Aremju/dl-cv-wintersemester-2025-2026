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

def select_ordered_keys(runs):
    # Prefer explicit EfficientNet model order
    desired = ['efficientnet-b1', 'efficientnet-b4', 'efficientnet-b7']
    present = [k for k in desired if k in runs]
    if present:
        return present
    # Fallback: heuristic by suffix b1/b4/b7 inside key
    alt_map = {'b1': None, 'b4': None, 'b7': None}
    for k in runs:
        for tag in alt_map:
            if tag in k.lower():
                alt_map[tag] = k
    ordered = [alt_map[t] for t in ['b1', 'b4', 'b7'] if alt_map[t]]
    return ordered

def plot_separate_loss_accuracy(runs, out_dir):
    keys = select_ordered_keys(runs)
    if len(keys) == 0:
        print("No matching runs for B1/B4/B7")
        return

    # Collect data
    plot_data = []
    titles = []
    for k in keys:
        paths = runs[k]
        train_p = paths.get('train')
        val_p = paths.get('val')
        rep_p = paths.get('report')
        train_json = load_json(train_p) if train_p and os.path.exists(train_p) else None
        val_json = load_json(val_p) if val_p and os.path.exists(val_p) else None
        report = load_json(rep_p) if rep_p and os.path.exists(rep_p) else None
        epochs, t_loss, t_acc, v_loss, v_acc = extract_metrics(train_json, val_json)
        if not epochs:
            print(f"Skipping {k}: no training data")
            continue
        title = k
        if report and isinstance(report, dict):
            model = report.get('model_name')
            if model:
                title = format_model_name(model)
        plot_data.append((epochs, t_loss, v_loss, t_acc, v_acc))
        titles.append(title)

    if not plot_data:
        print("No data to plot")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Loss figure
    fig_loss, axes_loss = plt.subplots(nrows=1, ncols=len(plot_data), figsize=(4 * len(plot_data), 4))
    if len(plot_data) == 1:
        axes_loss = [axes_loss]
    for ax, (epochs, t_loss, v_loss, _, _), title in zip(axes_loss, plot_data, titles):
        ax.plot(epochs, t_loss, '-', color='C0', linewidth=2, label='train loss')
        if v_loss:
            ax.plot(epochs[:len(v_loss)], v_loss, '-', color='C1', linewidth=2, label='val loss')
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')
    fig_loss.tight_layout()
    out_loss = os.path.join(out_dir, 'efficientnet_b1_b4_b7_loss.png')
    fig_loss.savefig(out_loss, dpi=200)
    plt.close(fig_loss)
    print(f"Saved loss plot: {out_loss}")

    # Accuracy figure
    fig_acc, axes_acc = plt.subplots(nrows=1, ncols=len(plot_data), figsize=(4 * len(plot_data), 4))
    if len(plot_data) == 1:
        axes_acc = [axes_acc]
    for ax, (epochs, _, _, t_acc, v_acc), title in zip(axes_acc, plot_data, titles):
        ax.plot(epochs, t_acc, '-', color='C2', linewidth=2, label='train acc')
        if v_acc:
            ax.plot(epochs[:len(v_acc)], v_acc, '-', color='C3', linewidth=2, label='val acc')
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right')
    fig_acc.tight_layout()
    out_acc = os.path.join(out_dir, 'efficientnet_b1_b4_b7_accuracy.png')
    fig_acc.savefig(out_acc, dpi=200)
    plt.close(fig_acc)
    print(f"Saved accuracy plot: {out_acc}")

def main():
    logs = 'logs'
    out = 'logs/plots'
    runs = find_runs(logs)
    plot_separate_loss_accuracy(runs, out)

if __name__ == '__main__':
    main()
