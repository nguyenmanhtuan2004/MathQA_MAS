import os
import re
import ast
import io
import contextlib
import builtins
import signal

from typing import Optional
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import Client, traceable, evaluate


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out.")

class ProgramOfThoughtsPrompt:
    def __init__(self, model_name: str = "", model_provider: str = "", temperature: float = 0.0):
        load_dotenv()
        self.model_name = os.getenv("MODEL_NAME")
        self.model_provider = os.getenv("MODEL_PROVIDER")
        self.temperature = os.getenv("TEMPERATURE")
        self.model = init_chat_model(self.model_name, model_provider=self.model_provider, temperature=self.temperature)

    def solve(self, question: str, context: Optional[str], show_code: bool):
        context_str = f"# Context:\n{context}\n" 
        pot_messages = [
        SystemMessage("You will write python program to solve math problems."),
        HumanMessage(content=f"""

        Examle 1:
        <QUESTION> 
        Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
        <ANSWER>
        ```python
        # Step 1: Set the initial number of toys Shawn has
        toys_initial = 5

        # Step 2: Set the number of toys received from mom
        mom_toys = 2

        # Step 3: Set the number of toys received from dad
        dad_toys = 2

        # Step 4: Calculate the total number of toys received
        total_received = mom_toys + dad_toys

        # Step 5: Calculate the total number of toys Shawn has now
        total_toys = toys_initial + total_received
        result = total_toys
        ```

        Example 2:
        <QUESTION>
        Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
        <ANSWER>
        ```python
        # Step 1: Set initial number of golf balls
        golf_balls_initial = 58

        # Step 2: Set number of golf balls lost on Tuesday
        golf_balls_lost_tuesday = 23

        # Step 3: Set number of golf balls lost on Wednesday
        golf_balls_lost_wednesday = 2

        # Step 4: Calculate remaining golf balls after both days
        golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
        result = golf_balls_left

        ```

        Example 3:
        <QUESTION> 
        There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
        <ANSWER>
        ```python
        # Step 1: Set the initial number of computers
        computers_initial = 9

        # Step 2: Set the number of computers installed each day
        computers_per_day = 5

        # Step 3: Set the number of days computers were installed (Monday to Thursday)
        num_days = 4

        # Step 4: Calculate the total number of computers added
        computers_added = computers_per_day * num_days

        # Step 5: Calculate the total number of computers now in the server room
        computers_total = computers_initial + computers_added
        result = computers_total
        ```

        # Include a final answer as a single number, no units or symbols.
        # For each step, provide a very brief explanation in one short sentence only.
        # The final answer 'MUST' be assigned the variable 'result'.
        # If the question includes time points, pay attention to time formats.
        # Before returning the final result, DOUBLE-CHECK each variable assignment and calculation to ensure they match the problem statement.
        {context_str}
        # Question: {question}
        """)]
        model_invoke=self.model.invoke(pot_messages)
        code = self.extract_code_from_markdown(model_invoke.content)

        if show_code:
            print("Generating code...")
            print("-" * 50)
            print("\n" + code + "\n")
        return code

    def safe_execute(self, code: str, timeout: int = 2):
        # Chỉ cho phép một số builtins an toàn
        safe_builtins = {
            "abs": abs, "min": min, "max": max, "sum": sum, "range": range, "len": len,
            "float": float, "int": int, "str": str, "bool": bool, "list": list, "dict": dict,
            "set": set, "tuple": tuple, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter
        }
        exec_globals = {"__builtins__": safe_builtins}
        exec_locals = {}

        # Đặt timeout cho code thực thi
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            exec(code, exec_globals, exec_locals)
            result = exec_locals.get("result", None)
            success = result is not None
        except TimeoutException as te:
            result = f"Timeout: {te}"
            success = False
        except Exception as e:
            result = f"Error: {e}"
            success = False
        finally:
            signal.alarm(0)  # Tắt alarm

        return result, success

    # Thay thế hàm execute cũ bằng safe_execute
    def execute(self, code: str):
        return self.safe_execute(code)

    def extract_code_from_markdown(self, text):
        # Tìm tất cả các đoạn code giữa ```python và ```
        code_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        # Gộp các đoạn code lại, cách nhau bởi 2 dòng trống
        return "\n\n".join(block.strip() for block in code_blocks)


    def check_syntax_and_logic(self, code_str):
        # Kiểm tra cú pháp
        try:
            ast.parse(code_str)
        except SyntaxError as e:
            print(f"\nError found: {e}\n")
            print('-' * 50)
            return f"Error: {e}"

        # Kiểm tra lỗi logic khi chạy
        try:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code_str, {})
            return None
        except Exception as ex:
            print(f"\nError found: {ex}\n")
            print('-' * 50)
            return f"Error: {ex}"

    def fix_error(self, code_str, error):
        fix_messages = [
        SystemMessage(content="""
            You are a professional programming assistant who analyzes and fixes code errors.
            You will be given a piece of code and an error message. 
            Your task is to:
            1. Analyze the cause of the error.
            2. Provide a corrected and working version of the code.
            """),
        HumanMessage(content=f"""
        # Code:
        {code_str}
        # Error: {error}
        """)]
        model_invoke=self.model.invoke(fix_messages)
        code = self.extract_code_from_markdown(model_invoke.content)
        print(f"\nFixed code:\n{code}\n")
        return code
