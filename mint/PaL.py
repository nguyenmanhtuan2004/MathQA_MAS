import os
import re

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from few_shot_PaL import few_shot_gsm8k


class ProgramAidedLanguagePrompt:
    def __init__(self, model_name: str = "", model_provider: str = "", temperature: float = 0.0):
        load_dotenv()
        self.model_name = os.getenv("MODEL_NAME")
        self.model_provider = os.getenv("MODEL_PROVIDER")
        self.temperature = os.getenv("TEMPERATURE")
        self.model = init_chat_model(self.model_name, model_provider=self.model_provider, temperature=self.temperature)

    def solve(self, question: str):
        pal_messages = [
            SystemMessage("You will write python program to solve math problems. You will only write code blocks."),
            HumanMessage(content=f"""
            {few_shot_gsm8k}
            # Answer this question by implementing a solver() function.
            # Include a final answer as a single number, no units or symbols.
            # 'CALL' the solver() function and then 'MUST' assign the variable 'result'.
            # If the question includes time points, pay attention to time formats.
            # Before returning the final result, DOUBLE-CHECK each variable assignment and calculation to ensure they match the problem statement.

            # Question: {question}
            """)]

        model_invoke = self.model.invoke(pal_messages)
        code = self.extract_code_from_markdown(model_invoke.content)
        return str(code)
    
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

    @staticmethod
    def exec_node(code: str):
        try:
            exec_globals = {}
            exec(code, exec_globals)
            result = exec_globals.get("result", None)

            if result is None:
                return str(9999)
            return str(result)
        except Exception as e:
            return str(9999)