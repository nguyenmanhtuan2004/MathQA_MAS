# MathQA_MAS - Math Question Answering_Multi Agent System

An intelligent math question answering system utilizing various AI methods with logical reasoning and step-by-step problem-solving capabilities.

## 🎯 Objectives

Develop a multi-agent system to answer math questions with a focus on:
- **Logical reasoning**: Analyze and understand the essence of the problem
- **Step-by-step solutions**: Provide detailed, easy-to-understand explanations
- **High accuracy**: Ensure correct results for various types of math problems
- **Versatile support**: Serve students, teachers, and math enthusiasts

## 🔬 Supported Prompting Methods

The system implements 5 advanced prompting methods:

1. **Zero-shot** - Direct answering without sample examples
2. **CoT (Chain of Thought)** - Step-by-step logical reasoning
3. **PoT (Program of Thoughts)** - Generate Python code for computation
4. **PaL (Program-aided Language)** - Combine natural language and programming
5. **MultiAgent** - Multiple agents collaborate to solve problems

## 📊 Datasets

The project uses 3 international benchmark datasets:

- **[GSM8K](https://github.com/openai/grade-school-math)** - 8,500 high-quality math problems for elementary students
- **[TATQA](https://github.com/NExTplusplus/TAT-QA)** - 16,552 financial report questions from 2,757 real-world contexts
- **[TABMWP](https://github.com/lupantech/PromptPG)** - 38,431 problems requiring reasoning over both text and tables

## 🚀 Installation & Setup

### System Requirements
- Python 3.8+
- OpenAI API key (or another LLM provider's API key)

### 1. Clone the repository

```bash
git clone https://github.com/qt1503/MAS.git
cd MAS
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp env.example .env
```

Edit the `.env` file with your information:

```env
# OpenAI API Configuration  
OPENAI_API_KEY="your-openai-api-key-here"

# LangSmith Tracing (optional for advanced tracking)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="your-langsmith-api-key-here"
LANGSMITH_PROJECT="your-project-name"

# Model Configuration
MODEL_NAME="gpt-4"
MODEL_PROVIDER="openai"
TEMPERATURE=0.2
```

## 💻 Usage

### Run single experiments

```bash
# Run few-shot PoT on a specific dataset
python few_shot_PoT.py

# Run a specific method via the mathqa module
python mathqa.py
```

### Evaluate on datasets

First, create the dataset on LangSmith:

```bash
# Create a specific dataset
python mathqa.py create-dataset --dataset GSM8K --limit 100

# Create all datasets
python mathqa.py create-dataset --dataset all --limit 50
```

Then run evaluation:

```bash
# Evaluate with Zero-shot method
python mathqa.py test --method Zero_shot --dataset GSM8K

# Evaluate with CoT method
python mathqa.py test --method CoT --dataset TATQA
```

### List available datasets

```bash
python mathqa.py datasets
```

### Generate analysis charts

```bash
# Generate comparison charts for methods
python __main__.py --metric comparison --methods Zero-shot CoT PoT PaL --datasets GSM8K TATQA --save

# Generate cost charts
python __main__.py --metric cost --methods CoT PoT --datasets GSM8K --save
```

## 📁 Project Structure

```
MAS/
├── __main__.py              # Main script for visualization and analysis
├── mathqa.py                # Utility module for mathematical operations  
├── few_shot_PoT.py          # Few-shot Program of Thoughts implementation
├── requirements.txt         # Required dependencies
├── env.example              # Environment configuration template
├── README_DATASET.md        # Detailed dataset documentation
├── datasets/                # Raw datasets
│   ├── GSM8K/               # Grade School Math dataset
│   │   ├── test.jsonl
│   │   ├── train.jsonl
│   │   ├── test_socratic.jsonl
│   │   ├── train_socratic.jsonl
│   │   └── example_model_solutions.jsonl
│   ├── TATQA/               # Table and Text QA dataset  
│   │   ├── tatqa_dataset_train.json
│   │   ├── tatqa_dataset_dev.json
│   │   ├── tatqa_dataset_test.json
│   │   └── tatqa_dataset_test_gold.json
│   └── TABMWP/              # Tabular Math Word Problems
│       ├── algorithm.png
│       ├── dataset.png
│       ├── prediction.png
│       ├── promptpg.png
│       └── tabmwp/
├── mint/                    # Core framework package
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management
│   ├── Zero_shot.py         # Zero-shot implementation
│   ├── CoT.py               # Chain of Thought implementation  
│   ├── PoT.py               # Program of Thoughts implementation
│   ├── PaL.py               # Program-aided Language implementation
│   ├── chart.py             # Chart generation utilities
│   ├── dataset_to_langsmith.py  # LangSmith integration tools
│   └── testing/             # Testing modules
│       └── CoT.py
├── EDA/                     # Exploratory Data Analysis notebooks
│   ├── GSM8K_EDA.ipynb
│   ├── TATQA_EDA.ipynb
│   └── TABMWP_EDA.ipynb
├── results/                 # Processed results from experiments
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
└── save_log/                # Detailed logs with timestamps
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

## 🔧 Advanced Features

### Sandbox Code Execution
The system uses a secure sandbox environment to execute Python code:
- **Isolated Environment**: Code runs in an isolated environment for system safety
- **Timeout Protection**: Execution time limits to prevent infinite loops
- **Resource Limiting**: Controls memory and CPU usage
- **Safe Libraries**: Only allows importing safe math/science libraries
- **Error Recovery**: Automatically detects and fixes syntax errors with a retry mechanism

### Advanced Visualization
The framework provides a powerful visualization system:
- **Dual-axis plots**: Display accuracy and other metrics simultaneously
- **Multi-dataset comparison**: Compare performance across multiple datasets
- **Custom color schemes**: Clearly distinguish methods with highlight colors
- **Statistical annotations**: Show exact values on charts
- **Publication-ready charts**: Export high-quality visualizations

### LangSmith Integration
Comprehensive integration with the LangSmith ecosystem:
- **Experiment Tracking**: Automatically logs all runs with metadata
- **Cost Monitoring**: Real-time tracking of API call costs
- **Performance Analytics**: Analyze accuracy and latency for each method
- **Dataset Versioning**: Manage versions of datasets and results
- **Traceability**: Full tracing from input to output for debugging

### Error Handling & Debugging
Intelligent debugging system with multiple layers:
- **Syntax Error Detection**: Automatically detects and fixes Python syntax errors
- **Logic Error Recovery**: Attempts to fix logical errors in generated code
- **Retry Mechanism**: Limits debug attempts to avoid infinite loops
- **Detailed Logging**: Detailed logs for each step to analyze issues
- **Graceful Degradation**: Fallback strategies when code execution fails

## 📊 Performance Evaluation

The system is evaluated on the following criteria:
- **Accuracy**: Correct answer rate
- **Cost**: API call cost
- **Latency**: Response time
- **Token Usage**: Number of tokens used

## 🔗 Contact

- GitHub: [@qt1503](https://github.com/qt1503), [@nguyenmanhtuan2004](https://github.com/nguyenmanhtuan2004)
- Project Link: [https://github.com/qt1503/MAS](https://github.com/qt1503/MAS)

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for GPT models
- [LangChain](https://langchain.com/) for the framework
- [LangSmith](https://smith.langchain.com/) for tracing and evaluation
- The authors of GSM8K, TATQA, and TABMWP datasets

