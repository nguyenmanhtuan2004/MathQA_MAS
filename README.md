# Math Question Answering

## 1. Purpose
The MathQA project aims to build a system for answering math questions, focusing on logical reasoning and step-by-step problem-solving, supporting students, educators, and math enthusiasts with accurate, clear, and detailed solutions for basic to advanced queries.

## 2. Objectives
Develop a multi-agent question-answering system focused on logical reasoning and step-by-step problem-solving, delivering accurate, clear, and detailed solutions for diverse mathematical queries.
## 3. Datasets
The following datasets are used in this project:
* [GSM8K](https://github.com/openai/grade-school-math)
* [TATQA](https://github.com/NExTplusplus/TAT-QA)
* [TABMWP](https://github.com/lupantech/PromptPG)


## 4. Install dependencies
To install all required libraries, simply run:

```
pip install -r requirements.txt
```

### Recommended: Use a virtual environment

It's recommended to use a Python virtual environment to avoid dependency conflicts:

```
python -m venv venv
source venv/bin/activate  

# On Windows: 
venv\Scripts\activate
```

### Environment setup

Copy the example environment file and update your API keys and settings:

```
cp env.example .env
```

Then, open the `.env` file and fill in the required values (API keys, model name, etc.) as described in the comments.

