from openai import OpenAI
from dateutil.relativedelta import relativedelta
import os
import json
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
import re
import langsmith as ls
from langsmith import traceable
from langsmith import Client, traceable, evaluate
from .preprocess_data import prepare_qa_input_with_answer_filter, standardize_item
from datetime import datetime

class MathReasoning(BaseModel):
    final_answer: str

class ZeroShotTesting:
    def __init__(self, model_name: str = "", model_provider: str = "", temperature: float = 0.0):
        load_dotenv()
        self.model_name = os.getenv("MODEL_NAME")
        self.model_provider = os.getenv("MODEL_PROVIDER")
        self.temperature = os.getenv("TEMPERATURE")
        self.model = init_chat_model(self.model_name, model_provider=self.model_provider, temperature=self.temperature)
        self.model_with_tools = self.model.with_structured_output(MathReasoning)
        self.all_results = []
        self.name = ""

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

        # Chuyển phân số sang số thập phân nếu có
        if '/' in raw_ans:
            try:
                numerator, denominator = raw_ans.split('/')
                decimal_value = float(numerator) / float(denominator)
                return str(decimal_value)
            except Exception:
                return raw_ans
        else:
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

    def testing(self, name: str):
        self.name = name.lower()
        self.all_results = []  # Reset results

        if self.name == "gsm8k":
            name_dataset = "GSM8K"
        elif self.name == "tatqa":
            name_dataset = "TATQA"
        else:
            name_dataset = "TABMWP"

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
                    - Include a `final_answer` as a single number, no units or symbols.
                    - If you cannot solve it, return a final_answer of "unknown".
                    - When dealing with money, do not round to thousands unless explicitly stated.
                """),
                HumanMessage(content=user_content)
            ]
            ai_msg = self.model_with_tools.invoke(messages)
            predicted_answer = ai_msg.final_answer
          
            return {
                "final_answer": predicted_answer,
                "question": question,
                "context": context
            }

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
                experiment_prefix=f"Zero-shot_{name_dataset}"
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

                with open(f"save_log/Zero-shot_results_{self.name}_{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}_{total}samples.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                    
                print(f"✅ Kết quả đã được lưu vào save_log/Zero-shot_results_{self.name}__{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}_{total}samples.json")
                print(f"📊 Độ chính xác: {accuracy:.2f}% ({correct}/{total})")
            else:
                print("❌ Không có kết quả nào được tạo ra. Vui lòng kiểm tra kết nối LangSmith và dataset.")
                
        except Exception as e:
            print(f"❌ Lỗi khi truy cập dataset hoặc chạy evaluation: {e}")
            return