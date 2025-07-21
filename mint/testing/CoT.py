from openai import OpenAI
from dateutil.relativedelta import relativedelta
import os
import json
from pydantic import BaseModel,Field
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI, init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import langsmith as ls
from langsmith import traceable, trace
from langsmith import Client, traceable, evaluate
from preprocess_data import prepare_qa_input_with_answer_filter
from datetime import datetime

file_path = '../dataset_langsmith/gsm8k.jsonl'

with open(file_path, "r", encoding="utf-8") as f:
    gsm8k = [json.loads(line) for line in f]


folder_path = "../dataset_langsmith/"
filename="tatqa.json"
file_path = os.path.join(folder_path, filename)
with open(file_path, "r", encoding="utf-8") as f:
    raw_data_tatqa = json.load(f)
tatqa = prepare_qa_input_with_answer_filter(raw_data_tatqa)


folder_path = "../dataset_langsmith/"
filename="tabmwp.json"
file_path = os.path.join(folder_path, filename)
with open(file_path, "r", encoding="utf-8") as f:
    tabmwp = json.load(f)

name= "gsm8k" # tatqa, tabmwp, gmm8k
length_test= 1
if name == "gsm8k":
   DATA=gsm8k 
   name_dataset="GSM8K"
elif name == "tatqa":
   DATA=tatqa
   name_dataset="TATQA"
else:
   DATA=tabmwp
   name_dataset="TABMWP"

load_dotenv()
model=init_chat_model('gpt-4o-mini',model_provider='openai',temperature=0.2)

class Step(BaseModel):
    explanation: str
    output: str
class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

model_with_tools = model.with_structured_output(MathReasoning)

def extract_ground_truth(answer, dataset_type):
    if dataset_type == "gsm8k":
        match = re.search(r"####\s*([\d,./]+)", answer)
        if match:
            raw_ans = match.group(1).replace(",", "").strip()
        else:
            raw_ans = answer.strip()
    elif dataset_type == "tatqa":
        if isinstance(answer, list):
            ans = str(answer[0]).strip()
        else:
            ans = str(answer).strip()
        ans = re.sub(r'^[\[\"]*([\d\-\.\/]+)[\]\"]*$', r'\1', ans)
        if '/' in ans:
            ans = re.sub(r"[^-\d/\.]", "", ans)
        else:
            ans = re.sub(r"[^-\d\.]", "", ans)
        raw_ans = ans
    else:
        raw_ans = str(answer).strip()
    return raw_ans


def compare_answers(predicted: str, actual: str, eps: float = 1e-2) -> bool:
    try:
        if '/' in predicted:
            try:
                numerator, denominator = predicted.split('/')
                predicted = str(float(numerator) / float(denominator))
            except Exception:
                predicted=str(predicted)
        if '/' in actual:
            try:
                numerator, denominator = actual.split('/')
                actual = str(float(numerator) / float(denominator))
            except Exception:
                actual=str(actual)
        pred = float(predicted.strip())
        act = float(actual.strip())
        return abs(pred - act) <= eps
    except ValueError:
        return predicted.strip().lower() == actual.strip().lower()

def unwrap_singleton(value):
    # Nếu là list hoặc tuple Python
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    # Nếu là chuỗi dạng '[2018]' hoặc "['2018']"
    if isinstance(value, str):
        import re
        match = re.fullmatch(r"\[\s*'?([-\w\.]+)'?\s*\]", value.strip())
        if match:
            return match.group(1)
    return value


@traceable(run_type="chain")
def target_function(inputs: dict):
    question = inputs["question"]
    context = inputs.get("context", "")
    # Nếu có context, nối vào trước question
    if context.strip():
        user_content = f"# Context:\n{context}\n\n# Question: {question}"
    else:
        user_content = question

    messages = [
        SystemMessage(content="""
        You are a math expert.
        For every question, you **must** respond using the `MathReasoning` tool.
        - Do not respond with plain text or natural language.
        - Use a list of `Step`s to break down the reasoning.
        - Include a `final_answer` as a single number, no units or symbols.
        """),
        HumanMessage(content=user_content)
    ]
    ai_msg = model_with_tools.invoke(messages)
    predicted_answer = ai_msg.final_answer  

    return {
        "final_answer": predicted_answer,
        "steps": getattr(ai_msg, "steps", None),
        "question": question,
        "context": context
    }
all_results = [] 
@traceable(run_type="tool")
def compare_result(inputs: dict, reference_outputs: dict, outputs: dict):
    predicted=str(unwrap_singleton(outputs["final_answer"]))
    actual = extract_ground_truth(str(reference_outputs["answer"]), f"{name}")
    eps = 1e-2
    try:
        if '/' in predicted:
            try:
                numerator, denominator = predicted.split('/')
                predicted = str(float(numerator) / float(denominator))
            except Exception:
                predicted = str(predicted)
        if '/' in actual:
            try:
                numerator, denominator = actual.split('/')
                actual = str(float(numerator) / float(denominator))
            except Exception:
                actual = str(actual)
        pred = float(predicted.strip())
        act = float(actual.strip())
        score = abs(pred - act) <= eps
    except ValueError:
        score = predicted.strip().lower() == actual.strip().lower()
    
    all_results.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": inputs["question"],
        "steps": [step.dict() for step in outputs["steps"]],
        "true_answer": actual,
        "predicted_answer": predicted,
        "context": inputs.get("context", ""),
        "correct": score
    })

    # Sau khi chạy xong:
    correct = sum(1 for x in all_results if x["correct"])
    total = len(all_results)
    accuracy = correct / total * 100
    wrong_answers = [x for x in all_results if not x["correct"]]

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "wrong_answers": wrong_answers
    }

    with open(f"save_log/CoT_results - {name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {"key": "is_correct", "score": int(score)}

client = Client()
evaluate(
    target_function,
    data=client.list_examples(dataset_name=f"{name_dataset}", splits=["base"]),
    evaluators=[compare_result],
    experiment_prefix=f"CoT_{name_dataset}"
)
