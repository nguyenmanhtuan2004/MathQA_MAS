import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

def load_data(method, dataset, folder="result"):
    file_path = os.path.join(folder, f"{method}_{dataset}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    return pd.read_csv(file_path)

def plot_bar_subplots(data_dict, metrics, methods, datasets, aspt):
    # Định nghĩa màu sắc nổi bật cho các methods
    highlight_colors = {
        'PaL': '#F28E2B',      
        'CoT': '#e74c3c',      
        'Zero-shot': '#76B7B2', 
        'PoT': '#3498db'       
    }
    # Tạo palette màu cho các method theo thứ tự
    custom_colors = [highlight_colors.get(method, '#DDA0DD') for method in methods]  # Mặc định tím nhạt

    titles = {
        "latency_sec": "Latency",      
        "total_cost": "Total Cost",                    
    }
    title_name = titles.get(aspt, '')

    for metric_col, display_name in metrics.items():
        fig, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=(6 * len(datasets), 5), sharey=True)
        if len(datasets) == 1:
            axs = [axs]
        
        # Tạo list để lưu các secondary axes
        axs2 = []
        # Thu thập tất cả dữ liệu để tính range chung cho y-axis
        all_values = []
        for method in methods:
            for dataset in datasets:
                key = (method, dataset)
                if key in data_dict:
                    df = data_dict[key]
                    if metric_col in df.columns:
                        all_values.append(df[metric_col].mean())

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
                        palette=custom_colors, edgecolor='black', alpha=0.85)

            for p in ax.patches:
                height = p.get_height()
                label = f"{height:.4f}" 
                ax.text(p.get_x() + p.get_width() / 2., height + 0.001 * height, label, ha="center", va="bottom")

            # Thêm line chart cho input tokens
            if aspt == 'tokens' and 'input_tokens' in data_dict[(methods[0], dataset)].columns and 'output_tokens' in data_dict[(methods[0], dataset)].columns:
                input_data = []
                output_data = []
                for method in methods:
                    key = (method, dataset)
                    if key in data_dict:
                        input_data.append(data_dict[key]['input_tokens'].mean())
                        output_data.append(data_dict[key]['output_tokens'].mean())
                    else:
                        input_data.append(0)
                        output_data.append(0)
                
                # Tạo secondary y-axis cho tokens
                ax2 = ax.twinx()
                # Vẽ 2 line charts
                ax2.plot(range(len(methods)), input_data, 'o-', color='red', linewidth=2, markersize=6, label='Input Tokens')
                ax2.plot(range(len(methods)), output_data, 's-', color='blue', linewidth=2, markersize=6, label='Output Tokens')
                ax2.legend(loc='upper right', fontsize=10, frameon=True)
                if i == 0:  # Chỉ hiển thị ylabel cho subplot đầu tiên
                    ax2.set_ylabel('Tokens', color='black')
                    ax2.tick_params(axis='y', labelcolor='black')
                else:
                    ax2.set_ylabel("")
                    ax2.tick_params(axis='y', labelright=False)
                axs2.append(ax2)
            elif aspt in data_dict[(methods[0], dataset)].columns:
                data = []
                for method in methods:
                    key = (method, dataset)
                    if key in data_dict and aspt in data_dict[key].columns:
                        mean_cost = data_dict[key][aspt].mean()
                        data.append(mean_cost)
                    else:
                        data.append(0)
                
                # Tạo secondary y-axis cho cost
                ax2 = ax.twinx()
                ax2.plot(range(len(methods)), data, 'o-', color='red', linewidth=2, markersize=6, label=title_name)
                ax2.legend(loc='upper right', fontsize=10, frameon=True)
                if i == 0:  # Chỉ hiển thị ylabel cho subplot đầu tiên
                    ax2.set_ylabel(title_name, color='black')
                    ax2.tick_params(axis='y', labelcolor='black')
                else:
                    ax2.set_ylabel("")
                    ax2.tick_params(axis='y', labelright=False)
                axs2.append(ax2)

            # Tạo chữ cái theo thứ tự A, B, C, ...
            letter = chr(65 + i)  # 65 là mã ASCII của 'A'
            ax.set_title(f"{letter} - {dataset}", fontweight="bold")
            if i == 0:  # Chỉ hiển thị ylabel cho subplot đầu tiên
                ax.set_ylabel(display_name)
            else:
                ax.set_ylabel("")  # Ẩn ylabel cho các subplot khác
            ax.set_xlabel("")  # Ẩn xlabel
            ax.set_xticklabels([])  # Ẩn labels trên trục x
            ax.grid(axis='y', linestyle='--', alpha=0.6, color='lightgray')
            ax.set_ylim(0.8, 1)  # Set giới hạn trục y từ 0 đến 0.8

        # Share y-axis cho secondary axes (cost)
        if axs2:
            # Thu thập tất cả giá trị cost để tính range chung
            all_values = []
            for method in methods:
                for dataset in datasets:
                    key = (method, dataset)
                    if key in data_dict and aspt in data_dict[key].columns:
                        all_values.append(data_dict[key][aspt].mean())
            
            if all_values:
                min_cost = min(all_values)
                max_cost = max(all_values)
                cost_range = max_cost - min_cost
                # Set cùng range cho tất cả secondary axes
                for ax2 in axs2:
                    ax2.set_ylim(min_cost - 0.1 * cost_range, max_cost + 0.1 * cost_range)

        # Tạo legend chung cho toàn bộ figure
        legend_handles = [Patch(color=custom_colors[i], label=method) for i, method in enumerate(methods)]
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=len(methods))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Tạo khoảng trống cho legend
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="So sánh nhiều kỹ thuật và dataset")
    parser.add_argument("--methods", nargs="+", required=True, help="Zero-shot CoT PoT")
    parser.add_argument("--datasets", nargs="+", required=True, help="GSM8K TATQA TABMWP")
    parser.add_argument("--metric",  required=True, choices=['tokens', 'latency_sec', 'total_cost'])
    args = parser.parse_args()

    methods = args.methods
    datasets = args.datasets

    metrics = {
        "is_correct": "Accuracy",
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
        print("Không có file hợp lệ nào được load. Kiểm tra lại tên file và thư mục `result/`.")
        return

    if missing:
        print("Một số file bị thiếu:")
        for file in missing:
            print(f" - {file}")

    plot_bar_subplots(data_dict, metrics, methods, datasets, args.metric)

if __name__ == "__main__":
    main()

