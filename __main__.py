import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def plot_bar(df_zero, df_cot):
    metrics = {
        "is_correct": "Accuracy",
        "total_cost": "Total Cost",
        "total_tokens": "Total Tokens",
        "latency_sec": "Latency (sec)"
    }

    for metric_col, display_name in metrics.items():
        if metric_col not in df_zero.columns or metric_col not in df_cot.columns:
            continue

        mean_zero = df_zero[metric_col].mean()
        mean_cot = df_cot[metric_col].mean()

        df_plot = pd.DataFrame({
            "method": ["Zero-shot", "CoT"],
            "value": [mean_zero, mean_cot]
        })

        plt.figure(figsize=(6, 4))
        ax = sns.barplot(data=df_plot, x="method", y="value", palette=["skyblue", "lightgreen"], edgecolor='black')

        for p in ax.patches:
            height = p.get_height()
            if metric_col == "total_cost":
                label = f"{height:.7f}"
            else:
                label = f"{height:.2f}"
            ax.text(p.get_x() + p.get_width() / 2., height + 0.01 * height, label, ha="center", va="bottom")

        plt.title(f"So sánh {display_name}", fontweight="bold")
        plt.ylabel(display_name)
        plt.ylim(0, max(mean_zero, mean_cot) * 1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

def plot_token_lines(df_zero, df_cot, max_points=100):
    metrics = ["input_tokens", "output_tokens"]

    df_zero = df_zero.copy()
    df_cot = df_cot.copy()
    df_zero["method"] = "Zero-shot"
    df_cot["method"] = "CoT"

    df_zero = df_zero.head(max_points).reset_index().rename(columns={"index": "sample_id"})
    df_cot = df_cot.head(max_points).reset_index().rename(columns={"index": "sample_id"})
    combined_df = pd.concat([df_zero, df_cot], ignore_index=True)

    # Vẽ từng metric
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=combined_df, x="sample_id", y=metric, hue="method", style="method", markers=False,linewidth=2)
        plt.title(f"{metric} theo từng run", fontweight="bold")
        plt.xlabel("Giá trị")
        plt.ylabel(metric)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()




def main():
    parser = argparse.ArgumentParser(description="So sánh kết quả Zero-shot và CoT")
    parser.add_argument("--zeroshot", required=True, help="File CSV kết quả Zero-shot")
    parser.add_argument("--cot", required=True, help="File CSV kết quả CoT")
    args = parser.parse_args()

    df_zero = load_data(args.zeroshot)
    df_cot = load_data(args.cot)
    plot_bar(df_zero, df_cot)
    plot_token_lines(df_zero, df_cot)

if __name__ == "__main__":
    main()
