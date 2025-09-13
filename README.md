# MathQA_MAS - Math Question Answering_Multi Agent System

An intelligent math question answering system utilizing various AI methods with logical reasoning and step-by-step problem-solving capabilities.

> **Note**: This repository is an enhanced version of the original [MathQA_MAS](https://github.com/nguyenmanhtuan2004/MathQA_MAS) project, which was collaboratively developed by [@qt1503](https://github.com/qt1503) and [@nguyenmanhtuan2004](https://github.com/nguyenmanhtuan2004). This version includes additional features such as a Streamlit web interface, enhanced multi-agent workflows, and improved data analysis capabilities.

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

### Dataset Organization
- **Raw datasets**: Original dataset files in their native formats
- **Filtered datasets**: Preprocessed and cleaned versions for evaluation
- **EDA notebooks**: Exploratory data analysis for each dataset
- **LangSmith integration**: Automatic dataset creation and management

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

**Key Dependencies:**
- **Streamlit**: Web interface framework
- **LangChain**: LLM framework and integrations
- **LangSmith**: Experiment tracking and evaluation
- **LangGraph**: Multi-agent workflow orchestration
- **OpenAI**: GPT model access
- **Pandas**: Data manipulation and analysis
- **Jupyter**: Interactive notebooks for analysis

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

### Web Interface (Streamlit)

Launch the interactive web interface to solve math questions:

```bash
streamlit run mathqa.py            
```

This will start a Streamlit application where you can:
- Enter math questions directly
- Select from 5 different solving methods
- View step-by-step solutions and generated code
- Get real-time results with an intuitive interface


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
# Fetch experiment data from LangSmith and generate analysis
jupyter notebook fetch_data.ipynb

# Generate comparison charts for methods  
python __main__.py --metric comparison --methods Zero-shot CoT PoT PaL --datasets GSM8K TATQA --save

# Generate cost charts
python __main__.py --metric cost --methods CoT PoT --datasets GSM8K --save
```

## 📁 Project Structure

```
MAS/
├── __main__.py              # Main script for visualization and analysis
├── mathqa.py                # Web interface and CLI utility for math question solving
├── few_shot_PoT.py          # Few-shot Program of Thoughts implementation
├── few_shot_PaL.py          # Few-shot Program-aided Language implementation  
├── requirements.txt         # Required dependencies
├── env.example              # Environment configuration template
├── README_DATASET.md        # Detailed dataset documentation
├── README_VISUALIZATION.md  # Visualization and analysis documentation
├── .streamlit/              # Streamlit configuration
│   └── config.toml          # Streamlit theme configuration
├── datasets/                # Raw datasets
│   ├── FILTER_DATASET/      # Filtered versions of datasets
│   │   ├── gsm8k.jsonl
│   │   ├── tabmwp.json
│   │   └── tatqa.json
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
│   ├── dataset_to_langsmith.py  # LangSmith integration tools
│   ├── fetch_data.ipynb         # Jupyter notebook for fetching and analyzing experiment data
│   └── testing/             # Testing modules and prompts
│       ├── CoT.py           # CoT testing implementation
│       ├── MultiAgent.py    # Multi-agent testing implementation
│       ├── PaL.py           # PaL testing implementation
│       ├── PoT.py           # PoT testing implementation
│       ├── Zero_shot.py     # Zero-shot testing implementation
│       ├── preprocess_data.py # Data preprocessing utilities
│       ├── test.py          # Main testing orchestration
│       └── prompts/         # Few-shot prompt templates
│           ├── few_shot_PaL.py
│           ├── few_shot_PoT.py
│           └── __pycache__/
├── EDA/                     # Exploratory Data Analysis notebooks
│   ├── GSM8K_EDA.ipynb
│   ├── TATQA_EDA.ipynb
│   └── TABMWP_EDA.ipynb
├── results                 # Processed results from experiments
└── save_log                # Detailed logs with timestamps

```

## 🔧 Advanced Features

### Interactive Web Interface
The system now includes a user-friendly Streamlit web interface:
- **Real-time Problem Solving**: Enter math questions and get instant solutions
- **Method Selection**: Choose from 5 different solving approaches with a dropdown
- **Step-by-step Visualization**: View detailed reasoning steps for CoT method
- **Code Display**: See generated Python code for PoT and PaL methods
- **Error Handling**: Graceful error handling with user-friendly messages
- **Responsive Design**: Clean, modern interface with gradient backgrounds

### Multi-Agent System
Advanced Multi-Agent workflow with specialized roles:
- **PreProcessing Agent**: Cleans and standardizes input questions
- **CodeGenerator Agent**: Generates Python code based on the question
- **Verifier Agent**: Checks syntax and logic of generated code
- **Executor Agent**: Safely runs code in a sandboxed environment
- **Debug_Feedback Agent**: Fixes errors with intelligent retry mechanism
- **Answer Agent**: Formats and presents the final result

### Sandbox Code Execution
The system uses a secure sandbox environment to execute Python code:
- **Thread-based Timeout**: Safe execution with configurable time limits
- **Isolated Environment**: Code runs in an isolated environment for system safety
- **Resource Limiting**: Controls memory and CPU usage
- **Safe Libraries**: Only allows importing safe math/science libraries
- **Error Recovery**: Automatically detects and fixes syntax errors with a retry mechanism

### Advanced Visualization
The framework provides a powerful visualization system:
- **Jupyter Integration**: Interactive notebooks for data exploration and analysis
- **Experiment Data Fetching**: Automated data retrieval from LangSmith experiments
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
- **Limited Retry Mechanism**: Prevents infinite loops with maximum 2 debug attempts
- **Detailed Logging**: Comprehensive logs for each step to analyze issues
- **Graceful Degradation**: Fallback strategies when code execution fails
- **User-friendly Messages**: Clear error messages displayed in the web interface

## 📊 Performance Evaluation

The system is evaluated on the following criteria:
- **Accuracy**: Correct answer rate across different methods
- **Cost**: API call cost analysis and optimization
- **Latency**: Response time measurement and comparison
- **Token Usage**: Input/output token consumption tracking
- **Error Rate**: Frequency and types of errors encountered
- **Debug Success**: Effectiveness of the automatic error correction system

### Evaluation Workflow
1. **Dataset Creation**: Automated dataset preparation on LangSmith
2. **Batch Evaluation**: Large-scale testing across multiple methods
3. **Real-time Feedback**: Immediate accuracy scoring during evaluation
4. **Result Analysis**: Comprehensive analysis with Jupyter notebooks
5. **Visualization**: Interactive charts and performance comparisons

## 🆕 What's New in This Fork

This enhanced version includes several major improvements over the original MathQA_MAS:

### 🖥️ **Web Interface**
- **Streamlit Application**: Interactive web interface for real-time problem solving
- **Method Selection**: Dropdown to choose between 5 different AI methods
- **Live Code Display**: View generated Python code for PoT and PaL methods
- **Step-by-step Visualization**: See detailed reasoning for CoT method

### 🤖 **Enhanced Multi-Agent System**
- **Improved Error Handling**: Smarter debugging with limited retry mechanism
- **Thread-based Execution**: Safer code execution with proper timeout handling
- **Better Agent Coordination**: Streamlined workflow between preprocessing, generation, verification, and execution

### 📊 **Advanced Analytics**
- **Jupyter Integration**: Interactive notebooks for comprehensive data analysis
- **Experiment Tracking**: Enhanced LangSmith integration for better experiment management
- **Data Fetching Tools**: Automated experiment data retrieval and analysis
- **Visualization Tools**: Improved charting and comparison capabilities

### 🛠️ **Technical Improvements**
- **Modern Dependencies**: Updated to latest versions of LangChain, Streamlit, and other libraries
- **Better Code Organization**: Cleaner separation of concerns with dedicated modules
- **Enhanced Security**: Improved sandbox execution with thread-based timeouts
- **Comprehensive Testing**: Extended testing modules for all prompting methods

## � Project History

This project has evolved through collaborative development:

1. **Original Development**: The foundational MathQA_MAS system was collaboratively built by [@qt1503](https://github.com/qt1503) and [@nguyenmanhtuan2004](https://github.com/nguyenmanhtuan2004), implementing core multi-agent functionality and mathematical reasoning capabilities.

2. **Enhanced Version**: This repository represents the continued development by [@qt1503](https://github.com/qt1503), adding modern web interfaces, improved user experience, and advanced analytics features while maintaining the robust core architecture.

## �🔗 Contact

- GitHub: [@qt1503](https://github.com/qt1503), [@nguyenmanhtuan2004](https://github.com/nguyenmanhtuan2004)

## 🙏 Acknowledgments

- **Original Project**: [MathQA_MAS](https://github.com/nguyenmanhtuan2004/MathQA_MAS) - Collaboratively developed by [@qt1503](https://github.com/qt1503) and [@nguyenmanhtuan2004](https://github.com/nguyenmanhtuan2004), serving as the foundation for this enhanced version
- [OpenAI](https://openai.com/) for GPT models
- [LangChain](https://langchain.com/) for the framework
- [LangSmith](https://smith.langchain.com/) for tracing and evaluation
- The authors of GSM8K, TATQA, and TABMWP datasets

