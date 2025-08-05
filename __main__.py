import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

def load_data(method, dataset, folder="results"):
    file_path = os.path.join(folder, f"{method}_{dataset}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    return pd.read_json(file_path)

def plot_bar_subplots(data_dict, metrics, methods, datasets, aspt):
    # Định nghĩa màu sắc nổi bật cho các methods
    highlight_colors = {
        'PaL': '#F28E2B', # Cam
        'CoT': '#e74c3c', # Đỏ
        'Zero-shot': '#76B7B2', # Xanh lục
        'PoT': '#3498db', # Xanh dương
        'MultiAgent': "#74CBA2", # Xanh lá   
    }
    # Tạo palette màu cho các method theo thứ tự
    custom_colors = [highlight_colors.get(method, '#DDA0DD') for method in methods]  # Mặc định tím nhạt

    titles = {
        "latency_sec": "Latency",      
        "total_cost": "Total Cost",                    
    }
    title_name = titles.get(aspt, '')

    for metric_col, display_name in metrics.items():
        fig, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=(6 * len(datasets), 6), sharey=True)
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

            sns.barplot(data=df_plot, x="method", y="value", hue="method", ax=ax,
                        palette=custom_colors, edgecolor='black', alpha=0.85, legend=False)

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
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.08), ncol=len(methods))
        
        # Chỉ sử dụng subplots_adjust để tránh conflict với tight_layout
        plt.subplots_adjust(bottom=0.18, top=0.92, left=0.08, right=0.92, wspace=0.3)
        plt.show()

def load_csv(dataset, folder="results"):
    file_path = os.path.join(folder, f"MultiAgent_{dataset}-RMVe.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    return pd.read_csv(file_path)


def main():
    
    parser = argparse.ArgumentParser()
    print('-' * 50)

    subparsers = parser.add_subparsers(dest='command', title='Danh sách lệnh được hỗ trợ', description="Chọn một trong các lệnh bên dưới để sử dụng")

    # Measure metrics
    metrics_parser = subparsers.add_parser('measure', help='Đo lường các metrics của các phương pháp prompting')
    metrics_parser.add_argument("--methods", "-mth", nargs="+", required=True, help="Các phương pháp prompting để sử dụng", choices=['Zero-shot', 'PoT', 'CoT',  'PaL', 'MultiAgent'])
    metrics_parser.add_argument("--datasets", "-d", nargs="+", required=True, help="Các tập dữ liệu để kiểm tra", choices=['GSM8K', 'TATQA', 'TABMWP'])
    metrics_parser.add_argument("--metric",  "-mtr", required=True, help="Metric để đo lường", choices=['tokens', 'latency_sec', 'total_cost'])

    # Remove agent
    remove_parser = subparsers.add_parser('remove', help='Loại bỏ tác tử Verifier khỏi hệ thống')
    remove_parser.add_argument("--datasets", "-d", nargs="+", required=True, help="Các tập dữ liệu để kiểm tra", choices=['GSM8K', 'TATQA', 'TABMWP', 'all'], default='all')
    remove_parser.add_argument("--metric",  "-m", required=True, help="Metric để đo lường", choices=['tokens', 'latency_sec', 'total_cost'])

    args = parser.parse_args()

    metrics = {
        "is_correct": "Accuracy",
    }
    try:
        if args.command == 'measure':
            methods = args.methods
            datasets = args.datasets

            data_dict = {}
            missing = []

            for method in methods:
                for dataset in datasets:
                    try:
                        df = load_data(method, dataset)
                        data_dict[(method, dataset)] = df
                    except FileNotFoundError:
                        missing.append(f"{method}_{dataset}.json")

            if not data_dict:
                print("Không có file hợp lệ nào được load. Kiểm tra lại tên file và thư mục `results/`.")
                return

            if missing:
                print("Một số file bị thiếu:")
                for file in missing:
                    print(f" - {file}")

            plot_bar_subplots(data_dict, metrics, methods, datasets, args.metric)

        elif args.command == 'remove':
            datasets = args.datasets
            if 'all' in datasets:
                datasets = ['GSM8K', 'TATQA', 'TABMWP']

            # Dictionary để lưu trữ tất cả các dataframe
            dataframes = {}
            
            for dataset in datasets:
                try:
                    df = load_csv(dataset)
                    dataframes[dataset] = df  # Lưu dataframe vào dictionary
                except FileNotFoundError:
                    print(f"Không tìm thấy file cho tập dữ liệu {dataset}. Vui lòng kiểm tra lại.")
            
            if not dataframes:
                print("Không có dữ liệu nào được load. Vui lòng kiểm tra lại tên file và thư mục `results/`.")
                return

            plot_bar_subplots(dataframes, metrics, 'MultiAgent', datasets, args.metric)

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\nNgười dùng đã ngắt quá trình")
        sys.exit(1)
    except Exception as e:
        print(f"\nLỗi: {e}")
        sys.exit(1)

    
if __name__ == "__main__":
    main()
