# MathQA_MAS - Math Question Answering_Multi Agent System

Hệ thống trả lời câu hỏi toán học thông minh sử dụng nhiều phương pháp AI khác nhau với khả năng suy luận logic và giải quyết vấn đề từng bước một cách chi tiết.

## 🎯 Mục tiêu

Phát triển hệ thống multi-agent để trả lời câu hỏi toán học với trọng tâm là:
- **Suy luận logic**: Phân tích và hiểu bản chất của bài toán
- **Giải quyết từng bước**: Cung cấp lời giải chi tiết, dễ hiểu
- **Độ chính xác cao**: Đảm bảo kết quả chính xác cho nhiều loại bài toán khác nhau
- **Hỗ trợ đa dạng**: Phục vụ học sinh, giáo viên và những người yêu thích toán học

## 🔬 Phương pháp Prompting được hỗ trợ

Hệ thống triển khai 5 phương pháp prompting tiên tiến:

1. **Zero-shot** - Trả lời trực tiếp không cần ví dụ mẫu
2. **CoT (Chain of Thought)** - Suy luận theo chuỗi logic từng bước  
3. **PoT (Program of Thoughts)** - Sinh code Python để tính toán
4. **PaL (Program-aided Language)** - Kết hợp ngôn ngữ tự nhiên và lập trình
5. **MultiAgent** - Sử dụng nhiều agent phối hợp giải quyết bài toán

## 📊 Bộ dữ liệu

Dự án sử dụng 3 bộ dữ liệu chuẩn quốc tế:

- **[GSM8K](https://github.com/openai/grade-school-math)** - 8,500 bài toán toán học chất lượng cao cho học sinh tiểu học
- **[TATQA](https://github.com/NExTplusplus/TAT-QA)** - 16,552 câu hỏi về báo cáo tài chính từ 2,757 ngữ cảnh thực tế
- **[TABMWP](https://github.com/lupantech/PromptPG)** - 38,431 bài toán yêu cầu lý luận trên cả văn bản và bảng biểu

## 🚀 Cài đặt và thiết lập

### Yêu cầu hệ thống
- Python 3.8+
- OpenAI API key (hoặc API key của nhà cung cấp LLM khác)

### 1. Clone repository

```bash
git clone https://github.com/qt1503/MAS.git
cd MAS
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Thiết lập biến môi trường

```bash
cp env.example .env
```

Chỉnh sửa file `.env` với thông tin cần thiết:

```env
# OpenAI API Configuration  
OPENAI_API_KEY="your-openai-api-key-here"

# LangSmith Tracing (optional cho advanced tracking)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="your-langsmith-api-key-here"
LANGSMITH_PROJECT="your-project-name"

# Model Configuration
MODEL_NAME="gpt-4"
MODEL_PROVIDER="openai"
TEMPERATURE=0.2
```


## 💻 Cách sử dụng

### Chạy thí nghiệm đơn lẻ

```bash
# Chạy few-shot PoT trên dataset cụ thể
python few_shot_PoT.py

# Chạy một phương pháp cụ thể qua mathqa module
python mathqa.py
```

### Kiểm tra trên bộ dữ liệu

Trước tiên, tạo dataset trên LangSmith:

```bash
# Tạo một dataset cụ thể
python mathqa.py create-dataset --dataset GSM8K --limit 100

# Tạo tất cả datasets
python mathqa.py create-dataset --dataset all --limit 50
```

Sau đó chạy kiểm tra:

```bash
# Kiểm tra với phương pháp Zero-shot
python mathqa.py test --method Zero_shot --dataset GSM8K

# Kiểm tra với phương pháp CoT
python mathqa.py test --method CoT --dataset TATQA
```

### Xem danh sách datasets

```bash
python mathqa.py datasets
```

### Tạo biểu đồ phân tích

```bash
# Tạo biểu đồ so sánh các phương pháp
python __main__.py --metric comparison --methods Zero-shot CoT PoT PaL --datasets GSM8K TATQA --save

# Tạo biểu đồ chi phí
python __main__.py --metric cost --methods CoT PoT --datasets GSM8K --save
```

## 📁 Cấu trúc dự án

```
MAS/
├── __main__.py              # Script chính để tạo visualization và phân tích
├── mathqa.py               # Module utilities cho mathematical operations  
├── few_shot_PoT.py         # Implementation few-shot Program of Thoughts
├── requirements.txt        # Dependencies cần thiết
├── env.example            # Template cấu hình môi trường
├── README_DATASET.md      # Documentation chi tiết về datasets
├── datasets/              # Raw datasets
│   ├── GSM8K/            # Grade School Math dataset
│   │   ├── test.jsonl
│   │   ├── train.jsonl
│   │   ├── test_socratic.jsonl
│   │   ├── train_socratic.jsonl
│   │   └── example_model_solutions.jsonl
│   ├── TATQA/            # Table and Text QA dataset  
│   │   ├── tatqa_dataset_train.json
│   │   ├── tatqa_dataset_dev.json
│   │   ├── tatqa_dataset_test.json
│   │   └── tatqa_dataset_test_gold.json
│   └── TABMWP/           # Tabular Math Word Problems
│       ├── algorithm.png
│       ├── dataset.png
│       ├── prediction.png
│       ├── promptpg.png
│       └── tabmwp/
├── mint/                 # Core framework package
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration management
│   ├── Zero_shot.py      # Zero-shot implementation
│   ├── CoT.py           # Chain of Thought implementation  
│   ├── PoT.py           # Program of Thoughts implementation
│   ├── PaL.py           # Program-aided Language implementation
│   ├── chart.py         # Chart generation utilities
│   ├── dataset_to_langsmith.py  # LangSmith integration tools
│   └── testing/         # Testing modules
│       └── CoT.py
├── EDA/                 # Exploratory Data Analysis notebooks
│   ├── GSM8K_EDA.ipynb
│   ├── TATQA_EDA.ipynb
│   └── TABMWP_EDA.ipynb
├── results/             # Processed results từ experiments
│   ├── CoT_GSM8K.json
│   ├── CoT_TABMWP.json
│   ├── CoT_TATQA.json
│   ├── MultiAgent_GSM8K.json
│   ├── MultiAgent_TABMWP.json
│   ├── MultiAgent_TATQA.json
│   ├── PaL_GSM8K.json
│   ├── PaL_TATQA.json
│   ├── PoT_GSM8K.json
│   ├── PoT_TABMWP.json
│   ├── PoT_TATQA.json
│   ├── Zero-shot_GSM8K.json
│   ├── Zero-shot_TABMWP.json
│   └── Zero-shot_TATQA.json
└── save_log/           # Chi tiết logs với timestamps
    ├── CoT_results_gsm8k_26-07-2025_16:36:03_500samples.json
    ├── CoT_results_tabmwp_26-07-2025_18:12:37_500samples.json
    ├── CoT_results_tatqa_26-07-2025_17:01:36_500samples.json
    ├── MultiAgent_results_gsm8k_27-07-2025_01:40:00_500samples.json
    ├── MultiAgent_results_tabmwp_27-07-2025_00:54:48_500samples.json
    ├── MultiAgent_results_tatqa_27-07-2025_02:32:28_500samples.json
    ├── PaL_results_gsm8k_26-07-2025_19:01:30_500samples.json
    ├── PaL_results_tabmwp_26-07-2025_18:33:35_500samples.json
    ├── PaL_results_tatqa_26-07-2025_19:18:41_500samples.json
    ├── PoT_results_gsm8k_26-07-2025_14:34:04_500samples.json
    ├── PoT_results_tabmwp_26-07-2025_15:37:25_500samples.json
    ├── PoT_results_tatqa_26-07-2025_14:52:34_500samples.json
    ├── Zero-shot_results_gsm8k_26-07-2025_13:26:24_500samples.json
    ├── Zero-shot_results_tabmwp_26-07-2025_15:45:06_500samples.json
    └── Zero-shot_results_tatqa_26-07-2025_13:36:44_500samples.json
```

## 🔧 Tính năng nâng cao

### Sandbox Code Execution
Hệ thống sử dụng môi trường sandbox an toàn để thực thi code Python:
- **Isolated Environment**: Code được chạy trong môi trường cô lập, đảm bảo an toàn hệ thống
- **Timeout Protection**: Giới hạn thời gian thực thi để tránh infinite loops
- **Resource Limiting**: Kiểm soát việc sử dụng memory và CPU
- **Safe Libraries**: Chỉ cho phép import các thư viện math/science an toàn
- **Error Recovery**: Tự động phát hiện và sửa lỗi syntax với retry mechanism

### Advanced Visualization
Framework cung cấp hệ thống visualization mạnh mẽ:
- **Dual-axis plots**: Hiển thị accuracy và metrics khác cùng lúc
- **Multi-dataset comparison**: So sánh hiệu suất trên nhiều datasets
- **Custom color schemes**: Phân biệt rõ ràng giữa các methods với highlight colors
- **Statistical annotations**: Hiển thị giá trị chính xác trên biểu đồ
- **Publication-ready charts**: Export high-quality visualizations

### LangSmith Integration
Tích hợp toàn diện với LangSmith ecosystem:
- **Experiment Tracking**: Tự động ghi log tất cả các lần chạy với metadata
- **Cost Monitoring**: Theo dõi chi phí API calls theo thời gian thực
- **Performance Analytics**: Phân tích độ chính xác và latency của từng phương pháp
- **Dataset Versioning**: Quản lý versions của datasets và results
- **Traceability**: Full tracing từ input đến output cho debugging

### Error Handling & Debugging
Hệ thống debug thông minh với nhiều layers:
- **Syntax Error Detection**: Tự động phát hiện và sửa lỗi Python syntax
- **Logic Error Recovery**: Attempt to fix logical errors in generated code
- **Retry Mechanism**: Giới hạn số lần debug để tránh vòng lặp vô hạn
- **Detailed Logging**: Log chi tiết cho mỗi bước để phân tích vấn đề
- **Graceful Degradation**: Fallback strategies khi code execution fails

## 📊 Đánh giá hiệu suất

Hệ thống được đánh giá trên các tiêu chí:
- **Accuracy**: Tỷ lệ trả lời đúng
- **Cost**: Chi phí API calls
- **Latency**: Thời gian phản hồi
- **Token Usage**: Số token sử dụng


## 🔗 Liên hệ

- GitHub: [@qt1503](https://github.com/qt1503), [@nguyenmanhtuan2004](https://github.com/nguyenmanhtuan2004)
- Project Link: [https://github.com/qt1503/MAS](https://github.com/qt1503/MAS)

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) cho GPT models
- [LangChain](https://langchain.com/) cho framework
- [LangSmith](https://smith.langchain.com/) cho tracing và evaluation
- Các tác giả của GSM8K, TATQA, và TABMWP datasets

