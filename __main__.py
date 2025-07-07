import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(method, dataset, folder="result"):
    file_path = os.path.join(folder, f"{method}_{dataset}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    return pd.read_csv(file_path)

def plot_bar_subplots(data_dict, metrics, methods, datasets):
    for metric_col, display_name in metrics.items():
        fig, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=(6 * len(datasets), 5))
        if len(datasets) == 1:
            axs = [axs]

        for i, dataset in enumerate(datasets):
            df_plot = []
            for method in methods:
                key = (method, dataset)
                if key in data_dict:
                    df = data_dict[key]
                    if metric_col in df.columns:
                        mean_val = df[metric_col].mean()
                        df_plot.append({"method": method, "value": mean_val})

            df_plot = pd.DataFrame(df_plot)
            ax = axs[i]
            if df_plot.empty:
                ax.set_title(f"{dataset} - Không có dữ liệu")
                continue

            sns.barplot(data=df_plot, x="method", y="value", ax=ax,
                        palette="pastel", edgecolor='black')

            for p in ax.patches:
                height = p.get_height()
                label = f"{height:.7f}" if metric_col == "total_cost" else f"{height:.2f}"
                ax.text(p.get_x() + p.get_width() / 2., height + 0.01 * height, label, ha="center", va="bottom")

            ax.set_title(f"{dataset} - {display_name}", fontweight="bold")
            ax.set_ylabel(display_name)
            ax.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

def plot_token_line_subplots(data_dict, methods, datasets, max_points=100):
    token_metrics = ["input_tokens", "output_tokens"]

    for metric in token_metrics:
        fig, axs = plt.subplots(nrows=len(datasets), figsize=(10, 5 * len(datasets)))
        if len(datasets) == 1:
            axs = [axs]

        for i, dataset in enumerate(datasets):
            combined_df = []
            for method in methods:
                key = (method, dataset)
                if key in data_dict and metric in data_dict[key].columns:
                    df = data_dict[key].copy()
                    df["method"] = method
                    df = df.head(max_points).reset_index().rename(columns={"index": "sample_id"})
                    combined_df.append(df)

            df_combined = pd.concat(combined_df, ignore_index=True) if combined_df else pd.DataFrame()

            ax = axs[i]
            if df_combined.empty:
                ax.set_title(f"{dataset} - Không có dữ liệu")
                continue

            sns.lineplot(data=df_combined, x="sample_id", y=metric, hue="method",
                         style="method", markers=False, linewidth=2, ax=ax)

            ax.set_title(f"{dataset} - {metric} theo từng run", fontweight="bold")
            ax.set_xlabel("Sample")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="So sánh nhiều kỹ thuật và dataset")
    parser.add_argument("--methods", nargs="+", required=True, help="Danh sách kỹ thuật (vd: Zero-shot CoT PoT)")
    parser.add_argument("--datasets", nargs="+", required=True, help="Danh sách dataset (vd: GSM8K TATQA TABMWP)")
    args = parser.parse_args()

    methods = args.methods
    datasets = args.datasets

    metrics = {
        "is_correct": "Accuracy",
        "total_cost": "Total Cost",
        "total_tokens": "Total Tokens",
        "latency_sec": "Latency (sec)"
    }

    data_dict = {}
    missing = []

    for method in methods:
        for dataset in datasets:
            try:
                df = load_data(method, dataset)
                data_dict[(method, dataset)] = df
            except FileNotFoundError:
                missing.append(f"{method}_{dataset}.csv")

    if not data_dict:
        print("❌ Không có file hợp lệ nào được load. Kiểm tra lại tên file và thư mục `result/`.")
        return

    if missing:
        print("⚠️ Một số file bị thiếu:")
        for file in missing:
            print(f" - {file}")

    plot_bar_subplots(data_dict, metrics, methods, datasets)
    plot_token_line_subplots(data_dict, methods, datasets)

if __name__ == "__main__":
    main()
