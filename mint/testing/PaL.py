from openai import OpenAI
import os
import json
from pydantic import BaseModel,Field
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm
import re
import textwrap
import traceback
from typing import List, Union
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langsmith import traceable
from langsmith import Client, traceable, evaluate
from langsmith import evaluate, Client
import langsmith as ls
from preprocess_data import prepare_qa_input_with_answer_filter
from datetime import datetime
from prompts.few_shot_PaL import few_shot_tabmwp,few_shot_tatqa,few_shot_gsm8k

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


name= "tatqa" # tatqa, tabmwp, gsm8k
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
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.2)

class State(TypedDict):
    question: str
    context: Optional[str]
    program: Optional[str]
    result: Optional[str]
    final_answer: Optional[str]
    error: Optional[str]

if name=="gsm8k":
    select_fewshot=few_shot_gsm8k
elif name=="tatqa":
    select_fewshot=few_shot_tatqa
else:
    select_fewshot=few_shot_tabmwp


def extract_code_from_markdown(text):
    """
    Trích xuất code Python từ chuỗi markdown có dạng ```python ... ```
    """
    # Dùng regex để tìm đoạn code giữa ```python và ```
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Nếu không có markdown, trả về nguyên text
    return text.strip()



@traceable(run_type="prompt")
def pot_node(state: State) -> State:
    context_str = f"# Context:\n{state['context']}\n" if state.get("context") else ""
    pot_messages = [
        SystemMessage(f"You will write python program to solve math problems. You will only write code blocks."),
        HumanMessage(content=f"""
{select_fewshot}
# Answer this question by implementing a solver() function.
# Include a final answer as a single number, no units or symbols.
# 'CALL' the solver() function and then 'MUST' assign the variale 'result'.
# If the question includes time points, pay attention to time formats.
# Before returning the final result, DOUBLE-CHECK each variable assignment and calculation to ensure they match the problem statement.
{context_str}
# Question: {state["question"]}
""")]

    model_invoke=model.invoke(pot_messages)
    code = extract_code_from_markdown(model_invoke.content)
    return {**state, "program": code}

def exec_node(state: State) -> State:
    try:
        exec_globals = {}
        exec(state["program"], exec_globals)
        result = exec_globals.get("result", None)

        if result is None:
            raise ValueError("Missing `result`")
        return {**state, "result": str(result), "error": None}
    except Exception as e:
        return {**state, "result": None, "error": str(e)}

def check_eos(state: State) -> bool:
    if state["error"] is None:
        return True
    else:
        return False
def write_final_answer_node(state:State)->State:

    if state["error"] is None:
        result=str(state["result"])
    else:
        result=str(9999)
    return {**state,"final_answer":result}


builder = StateGraph(State)
builder.add_node("PoT", pot_node)
builder.add_node("Exec", exec_node)

builder.add_node("write_final_answer",write_final_answer_node)

builder.set_entry_point("PoT")
builder.add_edge("PoT", "Exec")
builder.add_edge("Exec", "write_final_answer")
builder.add_edge("write_final_answer",END)
graph = builder.compile()



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



def run_graph(inputs: dict):
    # Chuẩn bị state đầu vào cho graph
    state = {
        "question": inputs["question"],
        "context": inputs.get("context", ""),
        "error": None,
    }
    # Chạy graph
    final_state = graph.invoke(state)
    

    result = {
        "final_answer": final_state.get("final_answer", ""),
        "program": final_state.get("program", ""),
        "response": final_state.get("response", "")
    }
    return  result
all_results = [] 
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
        "program": outputs["program"],
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

    with open(f"save_log/PaL_results - {name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {"key": "is_correct", "score": int(score)}
    

@traceable(run_type="chain")
def target_function(inputs: dict):
    result = run_graph(inputs)
    return result


client = Client()
evaluate(
    target_function,
    data=client.list_examples(dataset_name=f"{name_dataset}", splits=["base"]),
    evaluators=[compare_result],
    experiment_prefix=f"PaL_{name_dataset}"
)
