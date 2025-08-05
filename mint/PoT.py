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
from few_shot_PoT import few_shot_tatqa


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
        select_fewshot = few_shot_tatqa
        context_str = f"# Context:\n{context}\n" 
        pot_messages = [
        SystemMessage("You will write python program to solve math problems."),
        HumanMessage(content=f"""
                     
        {select_fewshot}
        
        # Context:       
        {context_str}

        # Question: {question}
        """)]
        model_invoke=self.model.invoke(pot_messages)
        code = self.extract_code_from_markdown(model_invoke.content)

        if show_code:
            print("Đang tạo mã nguồn...")
            print("-" * 50)
            print("\n" + code + "\n")
        return code

    def safe_execute(self, code: str, timeout: int = 5):
        """Thực thi code với sandbox an toàn"""
        
        # Blacklist các patterns nguy hiểm
        forbidden_patterns = [
            r'import\s+os', r'from\s+os', r'__import__',
            r'import\s+sys', r'from\s+sys',
            r'import\s+subprocess', r'from\s+subprocess',
            r'eval\s*\(', r'exec\s*\(',
            r'open\s*\(', r'file\s*\(',
            r'input\s*\(', r'raw_input\s*\(',
            r'compile\s*\(', r'globals\s*\(', r'locals\s*\(',
            r'getattr\s*\(', r'setattr\s*\(', r'hasattr\s*\(',
            r'delattr\s*\(', r'dir\s*\(',
            r'\.__class__\.__', r'\.__mro__', r'\.__subclasses__',  # Chặn attribute access bypass
        ]
        
        # Kiểm tra patterns nguy hiểm
        for pattern in forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return f"❌ Forbidden operation detected: {pattern}", False
        
        # Tạo safe import function chỉ cho phép các module an toàn
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            allowed_modules = {
                'math', 'random', 'datetime', 'decimal', 'fractions',
                '_strptime', 'time', 'calendar', 're', 'collections',
                'itertools', 'statistics'  # Dependencies for datetime and common math modules
            }
            if name in allowed_modules:
                return __import__(name, globals, locals, fromlist, level)
            else:
                raise ImportError(f"Module '{name}' is not allowed in sandbox")
        
        # Restricted builtins với safe import
        safe_builtins = {
            # Math operations
            "abs": abs, "min": min, "max": max, "sum": sum, "pow": pow,
            "round": round, "divmod": divmod,
            
            # Type conversions  
            "int": int, "float": float, "str": str, "bool": bool,
            
            # Collections
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "len": len, "range": range, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "sorted": sorted,
            "reversed": reversed,
            
            # Logic operations
            "all": all, "any": any,
            
            # I/O (safe)
            "print": print,
            
            # Safe import function
            "__import__": safe_import,
            
            # Pre-imported safe modules
            "math": __import__('math'),
            "datetime": __import__('datetime'),
            "collections": __import__('collections'),
            "itertools": __import__('itertools'),
        }
        
        exec_globals = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
        }
        exec_locals = {}
        
        # Capture output
        old_stdout = io.StringIO()

        try:
            # Set timeout (Unix only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            with contextlib.redirect_stdout(old_stdout):
                exec(code, exec_globals, exec_locals)
            
            result = exec_locals.get("result", None)
            output = old_stdout.getvalue()
            
            if result is not None:
                return str(result), True
            elif output:
                return output.strip(), True
            else:
                return "❌ Không tìm thấy biến 'result' sau khi thực thi", False
                
        except TimeoutException:
            return "❌ Code execution timed out", False
        except Exception as e:
            return f"❌ Lỗi: {e}", False
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Tắt alarm

    def extract_code_from_markdown(self, text):
        # Tìm tất cả các đoạn code giữa ```python và ```
        code_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        # Gộp các đoạn code lại, cách nhau bởi 2 dòng trống
        return "\n\n".join(block.strip() for block in code_blocks)


    def check_syntax_and_logic(self, code_str):
        # Kiểm tra code có rỗng không
        if not code_str or code_str.strip() == "":
            return "Empty code"
            
        # Debug: hiển thị code được kiểm tra
        print(f"🔍 Checking code:")
        print("-" * 30)
        print(code_str)
        print("-" * 30)
            
        # Kiểm tra cú pháp
        try:
            ast.parse(code_str)
        except SyntaxError as e:
            print(f"Lỗi: {e}\n")
            print('-' * 50)
            return f"Lỗi: {e}"

        # Kiểm tra logic cơ bản - code phải có biến result
        if 'result' not in code_str:
            print("Lỗi: Code phải chứa biến 'result'\n")
            print('-' * 50)
            return "Lỗi: Code phải chứa biến 'result'"
        
        # Kiểm tra các biến được sử dụng nhưng chưa được định nghĩa
        try:
            tree = ast.parse(code_str)
            
            # Thu thập tất cả các biến được định nghĩa
            defined_vars = set()
            used_vars = set()
            
            for node in ast.walk(tree):
                # Biến được gán trực tiếp
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_vars.add(target.id)
                        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                            # Unpacking assignment: a, b = something
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    defined_vars.add(elt.id)
                
                # Biến trong for loop
                elif isinstance(node, ast.For):
                    if isinstance(node.target, ast.Name):
                        defined_vars.add(node.target.id)
                    elif isinstance(node.target, ast.Tuple) or isinstance(node.target, ast.List):
                        for elt in node.target.elts:
                            if isinstance(elt, ast.Name):
                                defined_vars.add(elt.id)
                
                # Biến trong list/dict comprehensions
                elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                    for generator in node.generators:
                        if isinstance(generator.target, ast.Name):
                            defined_vars.add(generator.target.id)
                        elif isinstance(generator.target, ast.Tuple) or isinstance(generator.target, ast.List):
                            for elt in generator.target.elts:
                                if isinstance(elt, ast.Name):
                                    defined_vars.add(elt.id)
                
                # Biến trong with statement
                elif isinstance(node, ast.With):
                    for item in node.items:
                        if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                            defined_vars.add(item.optional_vars.id)
                
                # Function parameters và function names
                elif isinstance(node, ast.FunctionDef):
                    defined_vars.add(node.name)
                    for arg in node.args.args:
                        defined_vars.add(arg.arg)
                
                # Import statements
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        defined_vars.add(name.split('.')[0])  # Chỉ lấy phần đầu của module
                
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        if name != '*':  # Bỏ qua from module import *
                            defined_vars.add(name)
                
                # Biến được sử dụng (load context)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_vars.add(node.id)
            
            # Loại bỏ các built-in functions và modules
            builtins_and_modules = {
                'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter', 
                'sum', 'min', 'max', 'abs', 'round', 'all', 'any',
                'int', 'float', 'str', 'bool', 'list', 'dict', 'set', 'tuple',
                'math', 'datetime', 'gcd', 'sorted', 'reversed',
                # Math module functions
                'sqrt', 'sin', 'cos', 'tan', 'pi', 'e', 'ceil', 'floor',
                'log', 'log10', 'exp', 'factorial', 'degrees', 'radians',
                # Exception types
                'Exception', 'ValueError', 'TypeError', 'AttributeError',
                # Constants
                'True', 'False', 'None',
                # Collections và itertools
                'Counter', 'defaultdict', 'deque', 'OrderedDict',
                'combinations', 'permutations', 'product', 'cycle', 'chain',
                # Statistics functions
                'mean', 'median', 'mode', 'variance', 'stdev',
                # Common variable patterns that might be confused with builtins
                'data', 'values', 'items', 'keys'
            }
            
            # Tìm kiếm pattern định nghĩa biến trong string nếu AST không phát hiện được
            import re
            assignment_patterns = [
                r'^(\w+)\s*=\s*[^=]',  # Basic assignment at start of line
                r'\n(\w+)\s*=\s*[^=]',  # Basic assignment after newline
                r'for\s+(\w+)\s+in\s+',  # For loop variable
                r'(\w+),\s*(\w+)\s*=',  # Tuple unpacking (2 vars)
                r'(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*=',  # Triple unpacking
                r'(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*=',  # Quad unpacking
                r'with\s+.*\s+as\s+(\w+):',  # With statement
                r'def\s+(\w+)\s*\(',  # Function definition
                r'class\s+(\w+)\s*[\(:]',  # Class definition
                r'(\w+)\s*\+=',  # Augmented assignment
                r'(\w+)\s*-=',  # Augmented assignment
                r'(\w+)\s*\*=',  # Augmented assignment
                r'(\w+)\s*/=',  # Augmented assignment
            ]
            
            for pattern in assignment_patterns:
                matches = re.findall(pattern, code_str)
                for match in matches:
                    if isinstance(match, tuple):
                        for var in match:
                            if var and var.isidentifier():
                                defined_vars.add(var)
                    else:
                        if match and match.isidentifier():
                            defined_vars.add(match)
            
            undefined_vars = used_vars - defined_vars - builtins_and_modules
            
            if undefined_vars:
                print(f"⚠️ Biến chưa được định nghĩa: {', '.join(undefined_vars)}")
                print(f"🔍 Biến đã định nghĩa: {', '.join(sorted(defined_vars))}")
                print(f"🔍 Biến được sử dụng: {', '.join(sorted(used_vars))}")
                return f"Lỗi: Biến chưa được định nghĩa: {', '.join(undefined_vars)}"
            else:
                print(f"✅ Tất cả biến đã được định nghĩa đúng cách")
                
        except Exception as e:
            print(f"⚠️ Không thể phân tích biến: {e}")
            
        return None  # Không có lỗi

    def fix_error(self, code_str, error):
        fix_messages = [
        SystemMessage(content="""
            You are a professional programming assistant who analyzes and fixes code errors.
            You will be given a piece of code and an error message. 
            Your task is to:
            1. Analyze the cause of the error.
            2. Provide a corrected and working version of the code.
            3. Make sure ALL variables are properly defined before use.
            4. Include the complete, self-contained code solution.
            5. The code must have a 'result' variable at the end.
            
            Common issues to fix:
            - Variables used before definition: Define the variable first
            - Missing import statements: Add necessary imports
            - Incomplete code blocks: Include all necessary code
            - Missing variable assignments: Initialize all variables
            
            Example of fixing undefined variable error:
            BAD:
            result = frequency + max_frequency  # frequency not defined
            
            GOOD:
            frequency = data.count(value)  # Define frequency first
            max_frequency = max(frequencies)  # Define max_frequency first
            result = frequency + max_frequency
            """),
        HumanMessage(content=f"""
        # Error: {error}
        
        # Code with error:
        ```python
        {code_str}
        ```
        
        Please analyze the error and provide a COMPLETE, corrected version of the code that:
        1. Defines ALL variables before using them (especially variables mentioned in the error)
        2. Is syntactically correct and executable
        3. Solves the intended mathematical problem
        4. Has a 'result' variable at the end containing the final answer
        5. Includes all necessary imports and variable definitions
        
        Make sure to define these specific variables that are causing errors: {error.split(':')[-1] if ':' in error else error}
        """)]
        model_invoke=self.model.invoke(fix_messages)
        code = self.extract_code_from_markdown(model_invoke.content)
        print(f"\nSửa lỗi trong đoạn mã:")
        print("-" * 40)
        print(code)
        print("-" * 40)
        return code
