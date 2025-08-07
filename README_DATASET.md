# 📊 Dataset Guide - MathQA_MAS

## Overview

MathQA_MAS uses three specialized datasets for different types of math problems, ranging from basic arithmetic to complex financial analysis. Each dataset is designed to evaluate the reasoning and problem-solving capabilities of AI systems in real-world scenarios.

## 🎯 Supported Datasets

### 1. GSM8K (Grade School Math 8K)

**📝 Description:** A high-quality elementary school math dataset developed by OpenAI.

**📊 Statistics:**
- **Total questions:** 8,500
- **Training set:** 7,474 problems
- **Test set:** 1,319 problems
- **Problem types:** Basic arithmetic (addition, subtraction, multiplication, division, percentages, ratios)
- **Difficulty:** Grades 2-8

**🎯 Features:**
- Problems are written in natural language
- Answers include detailed step-by-step solutions
- Focus on basic arithmetic reasoning
- Suitable for evaluating logical problem-solving skills

**📁 File Structure:**
```
datasets/GSM8K/
├── train.jsonl                 # 7,474 training problems
├── test.jsonl                  # 1,319 test problems
├── train_socratic.jsonl        # Socratic version (training)
├── test_socratic.jsonl         # Socratic version (test)
└── example_model_solutions.jsonl # Example model solutions
```

**📄 Data Format:**
```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

---

### 2. TABMWP (Tabular Math Word Problems)

**📝 Description:** A dataset of math word problems requiring reasoning over both text and tables.

**📊 Statistics:**
- **Total questions:** 38,431
- **Training set:** ~35,000 problems
- **Development set:** ~1,000 problems
- **Test set:** ~2,000 problems
- **Number of tables:** 35,442
- **Problem types:** Data analysis, statistics, comparison

**🎯 Features:**
- Combines information from text and tables
- Requires data comprehension and analysis
- Real-world problems from business, science, and daily life
- Diverse formats: multiple-choice and open-ended

**📁 File Structure:**
```
datasets/TABMWP/
├── tabmwp/
│   ├── problems_train.json      # Training problems
│   ├── problems_dev.json        # Development problems
│   ├── problems_dev1k.json      # 1k development subset
│   ├── problems_test.json       # Test problems
│   ├── problems_test1k.json     # 1k test subset
│   ├── splits.json              # Dataset split info
│   └── tables/                  # Table images
│       ├── 1.png               # Table 1
│       ├── 2.png               # Table 2
│       └── ...                 # 35,442 other tables
├── algorithm.png               # Algorithm diagram
├── dataset.png                 # Dataset description
├── prediction.png              # Prediction example
└── promptpg.png                # Prompt engineering
```

**📄 Data Format:**
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

**📝 Description:** A dataset for complex financial report analysis and question answering.

**📊 Statistics:**
- **Total questions:** 16,552
- **Training set:** ~13,000 questions
- **Development set:** ~1,700 questions
- **Test set:** ~1,800 questions
- **Number of contexts:** 2,757 real-world financial reports
- **Question types:** Span extraction, counting, arithmetic, comparison

**🎯 Features:**
- Data sourced from real company financial reports
- Requires complex reasoning over tables and text
- Focuses on both quantitative and qualitative analysis
- Demands understanding of finance and accounting

**📁 File Structure:**
```
datasets/TATQA/
├── tatqa_dataset_train.json     # Training dataset
├── tatqa_dataset_dev.json       # Development dataset
├── tatqa_dataset_test.json      # Test dataset (no labels)
└── tatqa_dataset_test_gold.json # Test dataset with labels
```

**📄 Data Format:**
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

## 🔧 Dataset Usage

### Using the DatasetLoad Class

The project provides a `DatasetLoad` class for convenient loading and management of all datasets:

```python
from mint.dataset_to_langsmith import DatasetLoad

# Initialize DatasetLoad - automatically loads all datasets
dataset_loader = DatasetLoad()

# Get dataset by name
gsm8k_data = dataset_loader.get_dataset('gsm8k')
tabmwp_data = dataset_loader.get_dataset('tabmwp')
tatqa_data = dataset_loader.get_dataset('tatqa')

# Or get all datasets at once
all_datasets = dataset_loader.get_all_datasets()
```

### Usage Example

```python
from mint.dataset_to_langsmith import DatasetLoad

# Initialize dataset loader
loader = DatasetLoad()

# Print statistics
print(f"GSM8K: {len(loader.gsm8k)} problems")
print(f"TABMWP: {len(loader.tabmwp)} problems")
print(f"TATQA: {len(loader.tatqa)} questions")

# Get a sample from each dataset
gsm8k_sample = loader.get_dataset('gsm8k')[0]
tabmwp_sample = loader.get_dataset('tabmwp')[0]
tatqa_sample = loader.get_dataset('tatqa')[0]

print("GSM8K sample:", gsm8k_sample['question'])
print("TABMWP sample:", tabmwp_sample['question'])
print("TATQA sample:", tatqa_sample['question'])
```

### Manual Data Loading (Optional)

If you want to load data manually without using the class:

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

## 📈 Evaluation and Metrics

### GSM8K
- **Main metric:** Exact Match Accuracy
- **Method:** Compare the final numeric answer
- **Baseline:** ~20% (random), ~90% (GPT-4)

### TABMWP  
- **Main metric:** Accuracy
- **Method:** Exact match for answers
- **Baseline:** ~60% (state-of-the-art models)

### TATQA
- **Metrics:** 
  - Exact Match (EM)
  - F1 Score  
  - BLEU Score (for generation tasks)
- **Baseline:** ~50-70% EM depending on question type

## 🚀 Exploratory Data Analysis (EDA)

The project provides detailed EDA notebooks:

- **GSM8K_EDA.ipynb:** Analysis of difficulty distribution and problem types in GSM8K
- **TABMWP_EDA.ipynb:** Exploration of table structure and question types in TABMWP  
- **TATQA_EDA.ipynb:** Analysis of complexity and context distribution in TATQA

## 📚 References

1. **GSM8K:** [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
2. **TABMWP:** [Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning](https://arxiv.org/abs/2209.14610)  
3. **TATQA:** [TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance](https://arxiv.org/abs/2105.07624)

## ⚠️ Important Notes

1. **Licensing:** All datasets have their own licenses; please comply with their terms of use
2. **Storage:** Total size ~2GB, ensure sufficient disk space
3. **Encoding:** All files use UTF-8 encoding
4. **Preprocessing:** Some datasets require preprocessing before use
