from openai import OpenAI
import os
import re
import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langsmith import Client, traceable, evaluate
from .preprocess_data import prepare_qa_input_with_answer_filter
from datetime import datetime
from .prompts.few_shot_PaL import few_shot_tabmwp, few_shot_tatqa, few_shot_gsm8k

class State(TypedDict):
    question: str
    context: Optional[str]
    program: Optional[str]
    result: Optional[str]
    final_answer: Optional[str]
    error: Optional[str]

class PaLTesting:
    def __init__(self, model_name: str = "", model_provider: str = "", temperature: float = 0.0):
        load_dotenv()
        self.model_name = os.getenv("MODEL_NAME")
        self.model_provider = os.getenv("MODEL_PROVIDER")
        self.temperature = os.getenv("TEMPERATURE")
        self.model = init_chat_model(self.model_name, model_provider=self.model_provider, temperature=self.temperature)
        self.all_results = []
        self.name = ""
        self.select_fewshot = ""
        self.graph = None

    def setup_dataset(self, name: str):
        """Setup dataset-specific configurations"""
        self.name = name.lower()
        
        if self.name == "gsm8k":
            self.select_fewshot = few_shot_gsm8k
        elif self.name == "tatqa":
            self.select_fewshot = few_shot_tatqa
        else:
            self.select_fewshot = few_shot_tabmwp
            
        # Setup graph
        self._build_graph()

    @staticmethod
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

    def pot_node(self, state: State) -> State:
        context_str = f"# Context:\n{state['context']}\n" if state.get("context") else ""
        pot_messages = [
            SystemMessage("You will write python program to solve math problems. You will only write code blocks."),
            HumanMessage(content=f"""
{self.select_fewshot}
# Answer this question by implementing a solver() function.
# Include a final answer as a single number, no units or symbols.
# 'CALL' the solver() function and then 'MUST' assign the variable 'result'.
# If the question includes time points, pay attention to time formats.
# Before returning the final result, DOUBLE-CHECK each variable assignment and calculation to ensure they match the problem statement.
{context_str}
# Question: {state["question"]}
""")]

        model_invoke = self.model.invoke(pot_messages)
        code = self.extract_code_from_markdown(model_invoke.content)
        return {**state, "program": code}

    @staticmethod
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

    @staticmethod
    def write_final_answer_node(state: State) -> State:
        if state["error"] is None:
            result = str(state["result"])
        else:
            result = str(9999)
        return {**state, "final_answer": result}

    def _build_graph(self):
        """Build the LangGraph workflow"""
        builder = StateGraph(State)
        builder.add_node("PoT", self.pot_node)
        builder.add_node("Exec", self.exec_node)
        builder.add_node("write_final_answer", self.write_final_answer_node)

        builder.set_entry_point("PoT")
        builder.add_edge("PoT", "Exec")
        builder.add_edge("Exec", "write_final_answer")
        builder.add_edge("write_final_answer", END)
        self.graph = builder.compile()

    @staticmethod
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

    @staticmethod
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

    def run_graph(self, inputs: dict):
        # Chuẩn bị state đầu vào cho graph
        state = {
            "question": inputs["question"],
            "context": inputs.get("context", ""),
            "error": None,
        }
        # Chạy graph
        final_state = self.graph.invoke(state)
        
        result = {
            "final_answer": final_state.get("final_answer", ""),
            "program": final_state.get("program", ""),
            "response": final_state.get("response", "")
        }
        return result

    def testing(self, name: str):
        self.setup_dataset(name)
        self.all_results = []  # Reset results

        if self.name == "gsm8k":
            name_dataset = "GSM8K"
        elif self.name == "tatqa":
            name_dataset = "TATQA"
        else:
            name_dataset = "TABMWP"

        @traceable(run_type="chain")
        def target_function(inputs: dict):
            result = self.run_graph(inputs)
            return result

        @traceable(run_type="tool")
        def compare_result(inputs: dict, reference_outputs: dict, outputs: dict):
            predicted = str(self.unwrap_singleton(outputs["final_answer"]))
            actual = self.extract_ground_truth(str(reference_outputs["answer"]), self.name)
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
            
            self.all_results.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": inputs["question"],
                "program": outputs.get("program", ""),
                "true_answer": actual,
                "predicted_answer": predicted,
                "context": inputs.get("context", ""),
                "correct": score
            })

            return {"key": "is_correct", "score": int(score)}

        client = Client()
        
        try:
            # Get dataset examples to check if dataset exists and has data
            examples = list(client.list_examples(dataset_name=name_dataset))
            print(f"📋 Tìm thấy {len(examples)} mẫu trong dataset {name_dataset}")
            
            if len(examples) == 0:
                print(f"❌ Dataset {name_dataset} trống hoặc không tồn tại!")
                return
                
            evaluate(
                target_function,
                data=examples,
                evaluators=[compare_result],
                experiment_prefix=f"PaL_{name_dataset}"
            )
            
            # Sau khi chạy xong:
            correct = sum(1 for x in self.all_results if x["correct"])
            total = len(self.all_results)
            
            if total > 0:
                accuracy = correct / total * 100
                wrong_answers = [x for x in self.all_results if not x["correct"]]

                summary = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "wrong_answers": wrong_answers
                }

                # Create save_log directory if it doesn't exist
                os.makedirs("save_log", exist_ok=True)

                with open(f"save_log/PaL_results_{self.name}_{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}_{total}samples.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

                print(f"✅ Kết quả đã được lưu vào save_log/PaL_results_{self.name}_{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}_{total}samples.json")
                print(f"📊 Độ chính xác: {accuracy:.2f}% ({correct}/{total})")
            else:
                print("❌ Không có kết quả nào được tạo ra. Vui lòng kiểm tra kết nối LangSmith và dataset.")
                
        except Exception as e:
            print(f"❌ Lỗi khi truy cập dataset hoặc chạy evaluation: {e}")
            return
