from typing import List, Literal, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable, Client, evaluate
from dotenv import load_dotenv
from datetime import datetime
from .prompts.few_shot_PoT import few_shot_tatqa, few_shot_gsm8k, few_shot_tabmwp

import os
import json
import re
import uuid
import sys

# Add the mint directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from PoT import ProgramOfThoughtsPrompt

class State(TypedDict):
    question: str
    answer: Optional[str]
    context: Optional[str]
    error: Optional[str]
    debug_count: int  

class MultiAgentTesting:
    def __init__(self):
        load_dotenv()
        self.all_results = []
        self.name = ""
        self.prompt = ProgramOfThoughtsPrompt()
        self.current_debug_history = []
        
    def setup_dataset(self, name: str):
        """Setup dataset-specific configurations"""
        self.name = name.lower()
        if self.name == "gsm8k":
            self.select_fewshot = few_shot_gsm8k
        elif self.name == "tatqa":
            self.select_fewshot = few_shot_tatqa
        else:
            self.select_fewshot = few_shot_tabmwp

    def PreProcessing(self, state: State):
        state["question"] = re.sub(r'\s+', ' ', state["question"].strip())                      
        return {**state}

    def CodeGenerator(self, state: State):
        if state["error"] is None:
            generated_code = self.prompt.solve(state["question"], state["context"], self.select_fewshot)
            return {**state, "answer": generated_code}
        else:
            return{**state, "error": None}

    def Verifier(self, state: State):
        error = self.prompt.check_syntax_and_logic(state["answer"])
        return {**state, "error": error}

    def Executor(self, state: State):
        # Sử dụng prompt.safe_execute đã được cải thiện bảo mật
        result, success = self.prompt.safe_execute(str(state["answer"]))
        if success:
            return {**state, "answer": result}
        else:
            return {**state, "error": result}

    def Debug_Feedback(self, state: State):
        fixed_code = self.prompt.fix_error(state["answer"], state["error"])
        debug_step = {
            "error": state["error"],
            "fixed_code": fixed_code
        }
        self.current_debug_history.append(debug_step)
        return {**state, "debug_count": state.get('debug_count', 0) + 1, "answer": fixed_code}

    def Answer(self, state: State):
        debug_count = state.get("debug_count")
        if debug_count == 2 and state.get("error") is not None:
            return {**state, "answer": 9999}  # Return a default value if not fixed within two iterations.
        else:
            return {**state}

    def decide_error(self, state) -> Literal["Executor", "Debug_Feedback"]:
        error = state.get('error', None)
        debug_count = state.get('debug_count', 0)
        if debug_count >= 2:
            return "Executor"
        if error is None:
            return "Executor"
        return "Debug_Feedback"

    def decide_executor(self, state) -> Literal["Answer", "Debug_Feedback"]:
        error = state.get('error', None)
        debug_count = state.get('debug_count', 0)
        if debug_count >= 2:
            return "Answer"
        if error is None:
            return "Answer"
        return "Debug_Feedback"

    def build_graph(self):
        """Build the multi-agent graph"""
        builder = StateGraph(State)
        builder.add_node("PreProcessing", self.PreProcessing)
        builder.add_node("CodeGenerator", self.CodeGenerator)
        builder.add_node("Verifier", self.Verifier)
        builder.add_node("Executor", self.Executor)
        builder.add_node("Debug_Feedback", self.Debug_Feedback)
        builder.add_node("Answer", self.Answer)

        builder.add_edge(START, "PreProcessing")
        builder.add_edge("PreProcessing", "CodeGenerator")
        builder.add_edge("CodeGenerator", "Verifier")
        builder.add_conditional_edges("Verifier", self.decide_error)
        builder.add_conditional_edges("Executor", self.decide_executor)
        builder.add_edge("Debug_Feedback", "CodeGenerator")
        builder.add_edge("Answer", END)
        
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    def run_graph(self, inputs: dict):
        """Run the multi-agent system on a single question"""
        # Reset debug history cho run mới
        self.current_debug_history = []
        
        graph = self.build_graph()
        
        state = State(
            question=inputs["question"],
            context=inputs.get("context", ""),
            answer=None,
            error=None,
            debug_count=0,
        )

        final_state = graph.invoke(
            input=state,
            config={"configurable": {"thread_id": str(uuid.uuid4())}}
        )
        
        return {
            "final_answer": str(final_state.get("answer", "")),
            "debug": [{"Error": step["error"], "Fixed Code": step["fixed_code"]} for step in self.current_debug_history],
        }

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
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        if isinstance(value, str):
            import re
            match = re.fullmatch(r"\[\s*'?([-\w\.]+)'?\s*\]", value.strip())
            if match:
                return match.group(1)
        return value

    def testing(self, name: str):
        """Run testing on the specified dataset"""
        self.setup_dataset(name)
        self.all_results = []

        if self.name == "gsm8k":
            name_dataset = "GSM8K"
        elif self.name == "tatqa":
            name_dataset = "TATQA"
        else:
            name_dataset = "TABMWP"

        @traceable(run_type="chain", name="MultiAgent_System")
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
                "response": outputs.get("response", ""),
                "true_answer": actual,
                "predicted_answer": predicted,
                "context": inputs.get("context", ""),
                "correct": score,
                "debug_count": outputs.get("debug_count", 0)
            })

            return {"key": "is_correct", "score": int(score)}

        client = Client()
        
        try:
            examples = list(client.list_examples(dataset_name=name_dataset))
            print(f"📋 Tìm thấy {len(examples)} mẫu trong dataset {name_dataset}")
            
            if len(examples) == 0:
                print(f"❌ Dataset {name_dataset} trống hoặc không tồn tại!")
                return
                
            evaluate(
                target_function,
                data=examples,
                evaluators=[compare_result],
                experiment_prefix=f"MultiAgent_{name_dataset}"
            )
            
            correct = sum(1 for x in self.all_results if x["correct"])
            total = len(self.all_results)
            
            if total > 0:
                accuracy = correct / total * 100
                avg_debug_count = sum(x["debug_count"] for x in self.all_results) / total
                wrong_answers = [x for x in self.all_results if not x["correct"]]

                summary = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "average_debug_iterations": avg_debug_count,
                    "wrong_answers": wrong_answers
                }

                os.makedirs("save_log", exist_ok=True)

                with open(f"save_log/MultiAgent_results_{self.name}_{datetime.now().strftime('%d-%m-%Y %H%M%S')}_{total}samples.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                    
                print(f"✅ Kết quả đã được lưu vào save_log/MultiAgent_results_{self.name}_{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}_{total}samples.json")
                print(f"📊 Độ chính xác: {accuracy:.2f}% ({correct}/{total})")
                print(f"🔄 Số lần debug trung bình: {avg_debug_count:.2f}")
            else:
                print("❌ Không có kết quả nào được tạo ra. Vui lòng kiểm tra kết nối LangSmith và dataset.")
                
        except Exception as e:
            print(f"❌ Lỗi khi truy cập dataset hoặc chạy evaluation: {e}")
            return
