# 📊 Hướng dẫn Datasets - MathQA_MAS

## Tổng quan

Dự án MathQA_MAS sử dụng 3 bộ datasets chuyên dụng cho các loại bài toán toán học khác nhau, từ toán học cơ bản đến phân tích tài chính phức tạp. Mỗi dataset được thiết kế để đánh giá khả năng suy luận và giải quyết vấn đề của hệ thống AI trong các tình huống thực tế.

## 🎯 Datasets được hỗ trợ

### 1. GSM8K (Grade School Math 8K)

**📝 Mô tả**: Bộ dữ liệu toán học cấp tiểu học chất lượng cao được OpenAI phát triển

**📊 Thống kê**:
- **Tổng số câu hỏi**: 8,500 bài toán
- **Training set**: 7,474 bài toán
- **Test set**: 1,319 bài toán  
- **Loại bài toán**: Toán học cơ bản (cộng, trừ, nhân, chia, phần trăm, tỷ lệ)
- **Độ khó**: Cấp độ học sinh tiểu học (lớp 2-8)

**🎯 Đặc điểm**:
- Các bài toán được viết bằng ngôn ngữ tự nhiên
- Đáp án kèm theo lời giải chi tiết từng bước
- Tập trung vào khả năng suy luận số học cơ bản
- Phù hợp để đánh giá khả năng giải quyết vấn đề logic

**📁 Cấu trúc file**:
```
datasets/GSM8K/
├── train.jsonl                 # 7,474 bài toán training
├── test.jsonl                  # 1,319 bài toán test
├── train_socratic.jsonl        # Phiên bản với hướng dẫn Socratic
├── test_socratic.jsonl         # Test set với phương pháp Socratic
└── example_model_solutions.jsonl # Ví dụ lời giải mẫu
```

**📄 Format dữ liệu**:
```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

---

### 2. TABMWP (Tabular Math Word Problems)

**📝 Mô tả**: Bộ dữ liệu bài toán yêu cầu suy luận kết hợp giữa văn bản và bảng biểu

**📊 Thống kê**:
- **Tổng số câu hỏi**: 38,431 bài toán
- **Training set**: ~35,000 bài toán
- **Development set**: ~1,000 bài toán  
- **Test set**: ~2,000 bài toán
- **Số lượng bảng**: 35,442 bảng dữ liệu
- **Loại bài toán**: Phân tích dữ liệu, thống kê, so sánh

**🎯 Đặc điểm**:
- Kết hợp thông tin từ văn bản và bảng biểu
- Yêu cầu khả năng đọc hiểu và phân tích dữ liệu
- Các bài toán thực tế về kinh doanh, khoa học, đời sống
- Đa dạng về format: trắc nghiệm và tự luận

**📁 Cấu trúc file**:
```
datasets/TABMWP/
├── tabmwp/
│   ├── problems_train.json      # Bài toán training
│   ├── problems_dev.json        # Bài toán development  
│   ├── problems_dev1k.json      # Subset 1k development
│   ├── problems_test.json       # Bài toán test
│   ├── problems_test1k.json     # Subset 1k test
│   ├── splits.json              # Thông tin phân chia dataset
│   └── tables/                  # Thư mục chứa hình ảnh bảng
│       ├── 1.png               # Bảng số 1
│       ├── 2.png               # Bảng số 2
│       └── ...                 # 35,442 bảng khác
├── algorithm.png               # Sơ đồ thuật toán
├── dataset.png                 # Mô tả dataset
├── prediction.png              # Ví dụ dự đoán
└── promptpg.png               # Prompt engineering
```

**📄 Format dữ liệu**:
```json
{
  "1": {
    "question": "The members of the local garden club tallied the number of plants in each person's garden. How many gardens have at least 47 plants?",
    "choices": null,
    "answer": "13",
    "table_id": "1",
    "grade": "4",
    "subject": "Math",
    "topic": "Data and graphs"
  }
}
```

---

### 3. TATQA (TAT-QA Dataset)

**📝 Mô tả**: Bộ dữ liệu câu hỏi phân tích báo cáo tài chính phức tạp

**📊 Thống kê**:
- **Tổng số câu hỏi**: 16,552 câu hỏi
- **Training set**: ~13,000 câu hỏi
- **Development set**: ~1,700 câu hỏi
- **Test set**: ~1,800 câu hỏi  
- **Số ngữ cảnh**: 2,757 ngữ cảnh từ báo cáo tài chính thực tế
- **Loại câu hỏi**: Span extraction, counting, arithmetic, comparison

**🎯 Đặc điểm**:
- Dữ liệu từ báo cáo tài chính thực tế của các công ty
- Yêu cầu suy luận phức tạp trên bảng và văn bản
- Tập trung vào phân tích định lượng và định tính
- Đòi hỏi hiểu biết về tài chính và kế toán

**📁 Cấu trúc file**:
```
datasets/TATQA/
├── tatqa_dataset_train.json     # Dataset training
├── tatqa_dataset_dev.json       # Dataset development
├── tatqa_dataset_test.json      # Dataset test (không có label)
└── tatqa_dataset_test_gold.json # Dataset test với label
```

**📄 Format dữ liệu**:
```json
{
  "table": {
    "uid": "e78f8b29-6085-43de-b32f-be1a68641be3",
    "table": [
      ["", "2019 %", "2018 %", "2017 %"],
      ["Revenue", "100.0", "100.0", "100.0"],
      ["Cost of revenue", "45.2", "44.8", "45.1"]
    ]
  },
  "paragraphs": [
    {
      "uid": "para_1",
      "text": "Revenue increased by 15% in 2019 compared to 2018..."
    }
  ],
  "questions": [
    {
      "uid": "question_1",
      "question": "What was the percentage change in revenue from 2018 to 2019?",
      "answer": "15%",
      "answer_type": "span",
      "derivation": "..."
    }
  ]
}
```

## 🔧 Sử dụng Datasets

### Sử dụng Class DatasetLoad

Dự án cung cấp class `DatasetLoad` để load và quản lý tất cả datasets một cách tiện lợi:

```python
from mint.dataset_to_langsmith import DatasetLoad

# Khởi tạo DatasetLoad - tự động load tất cả datasets
dataset_loader = DatasetLoad()

# Lấy dataset theo tên
gsm8k_data = dataset_loader.get_dataset('gsm8k')
tabmwp_data = dataset_loader.get_dataset('tabmwp') 
tatqa_data = dataset_loader.get_dataset('tatqa')

# Hoặc lấy tất cả datasets cùng lúc
all_datasets = dataset_loader.get_all_datasets()
```

### Ví dụ sử dụng

```python
from mint.dataset_to_langsmith import DatasetLoad

# Khởi tạo dataset loader
loader = DatasetLoad()

# In thống kê về số lượng dữ liệu
print(f"GSM8K: {len(loader.gsm8k)} bài toán")
print(f"TABMWP: {len(loader.tabmwp)} bài toán") 
print(f"TATQA: {len(loader.tatqa)} câu hỏi")

# Lấy một sample từ mỗi dataset
gsm8k_sample = loader.get_dataset('gsm8k')[0]
tabmwp_sample = loader.get_dataset('tabmwp')[0]
tatqa_sample = loader.get_dataset('tatqa')[0]

print("GSM8K sample:", gsm8k_sample['question'])
print("TABMWP sample:", tabmwp_sample['question'])
print("TATQA sample:", tatqa_sample['question'])
```

### Load dữ liệu thủ công (tùy chọn)

Nếu bạn muốn load dữ liệu thủ công không qua class:

```python
import json
import os
from mint.config import DATA_DIR

# GSM8K
def load_gsm8k_manual():
    file_path = os.path.join(DATA_DIR('GSM8K'), 'test.jsonl')
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# TABMWP  
def load_tabmwp_manual():
    file_path = os.path.join(DATA_DIR('TABMWP/tabmwp'), 'problems_test.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# TATQA
def load_tatqa_manual():
    file_path = os.path.join(DATA_DIR('TATQA'), 'tatqa_dataset_train.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
```

## 📈 Đánh giá và Metrics

### GSM8K
- **Metric chính**: Exact Match Accuracy
- **Phương pháp**: So sánh đáp án số cuối cùng
- **Baseline**: ~20% (random), ~90% (GPT-4)

### TABMWP  
- **Metric chính**: Accuracy
- **Phương pháp**: Exact match cho đáp án
- **Baseline**: ~60% (state-of-the-art models)

### TATQA
- **Metrics**: 
  - Exact Match (EM)
  - F1 Score  
  - BLEU Score (cho generation tasks)
- **Baseline**: ~50-70% EM tùy theo loại câu hỏi

## 🚀 Exploratory Data Analysis (EDA)

Dự án cung cấp các notebook phân tích dữ liệu chi tiết:

- **GSM8K_EDA.ipynb**: Phân tích phân bố độ khó, loại bài toán GSM8K
- **TABMWP_EDA.ipynb**: Khám phá cấu trúc bảng, loại câu hỏi TABMWP  
- **TATQA_EDA.ipynb**: Phân tích độ phức tạp, phân bố ngữ cảnh TATQA

## 📚 Tài liệu tham khảo

1. **GSM8K**: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
2. **TABMWP**: [Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning](https://arxiv.org/abs/2209.14610)  
3. **TATQA**: [TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance](https://arxiv.org/abs/2105.07624)

## ⚠️ Lưu ý quan trọng

1. **Bản quyền**: Tất cả datasets đều có license riêng, vui lòng tuân thủ điều khoản sử dụng
2. **Dung lượng**: Tổng dung lượng ~2GB, đảm bảo đủ không gian lưu trữ
3. **Encoding**: Tất cả file đều sử dụng UTF-8 encoding
4. **Preprocessing**: Một số datasets cần preprocessing trước khi sử dụng
