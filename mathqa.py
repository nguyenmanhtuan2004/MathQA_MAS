from typing import Literal, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from mint.PoT import ProgramOfThoughtsPrompt
from mint.chart import ChartGenerate, ExperimentDataFetcher
from mint.dataset_to_langsmith import DatasetToLangsmith
from mint.testing.test import TestingDataset
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client
from dotenv import load_dotenv

import argparse
import re
import sys
import uuid
import os
import io
import contextlib
import signal


class State(TypedDict):
    question: str
    answer: Optional[str]
    context: Optional[str]
    error: Optional[str]
    debug_count: int  
    show_reasoning: bool

class IntermediateProgram(BaseModel):
    program: str

def single_question(question: str, context: Optional[str] = None, show_reasoning: bool = False):
    
    prompt = ProgramOfThoughtsPrompt()

    builder = StateGraph(State)
    builder.add_node("PreProcessing", PreProcessing)
    builder.add_node("CodeGenerator", lambda state: CodeGenerator(state, prompt))
    builder.add_node("Verifier", lambda state: Verifier(state, prompt))
    builder.add_node("Executor", lambda state: Executor(state, prompt))
    builder.add_node("Debug_Feedback", lambda state: Debug_Feedback(state, prompt))
    builder.add_node("Answer", Answer)

    builder.add_edge(START, "PreProcessing")
    builder.add_edge("PreProcessing", "CodeGenerator")
    builder.add_edge("CodeGenerator", "Verifier")
    builder.add_conditional_edges("Verifier", decide_error)
    builder.add_conditional_edges("Executor", decide_executor)
    builder.add_edge("Debug_Feedback", "CodeGenerator")
    builder.add_edge("Answer", END)
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    state = State(
        question=question,
        context=context,
        answer=None,
        error=None,
        debug_count=0,
        show_reasoning=show_reasoning
    )

    graph.invoke(
        input=state,
        config={"configurable": {"thread_id": str(uuid.uuid4())}}
    )


def PreProcessing(state: State):
    state["question"] = re.sub(r'\s+', ' ', state["question"].strip())                      
    print("Question: " + state["question"] + "\n")
    return {**state}

def CodeGenerator(state: State, prompt):
    generated_code = prompt.solve(state["question"], state["context"], state["show_reasoning"])
    return {**state, "debug_count": state.get('debug_count', 0), "answer": generated_code}

def Verifier(state: State, prompt):
    error = prompt.check_syntax_and_logic(state["answer"])
    return {**state, "debug_count": state.get('debug_count', 0), "error": error}

def Executor (state: State, prompt):
    result, success = prompt.safe_execute(str(state["answer"]))
    if success:
        return {**state, "debug_count": state.get('debug_count', 0), "answer": result}
    else:
        return {**state, "debug_count": state.get('debug_count', 0), "error": result}

def Debug_Feedback(state: State, prompt):
    fixed_code = prompt.fix_error(state["answer"], state["error"])
    return {**state, "debug_count": state.get('debug_count', 0) + 1, "error": None, "answer": fixed_code}

def Answer(state: State):
    answer = state.get("answer")
    if answer is not None:
        print("Answer: " + str(answer) + "\n")
    else:
        print("Answer: None\n")
    return {**state, "debug_count": state.get('debug_count', 0), "answer": answer}

def decide_error(state) -> Literal["Executor", "Debug_Feedback"]:
    error = state.get('error', None)
    debug_count = state.get('debug_count', 0)
    if debug_count >= 2:
        return "Executor"
    if error is None:
        return "Executor"
    return "Debug_Feedback"

def decide_executor(state) -> Literal["Answer", "Debug_Feedback"]:
    error = state.get('error', None)
    debug_count = state.get('debug_count', 0)
    if debug_count >= 2:
        return "Answer"
    if error is None:
        return "Answer"
    return "Debug_Feedback"


def check_dataset_exists(dataset_name: str) -> bool:
    """
    Kiểm tra xem dataset đã tồn tại trên LangSmith chưa.
    
    Args:
        dataset_name (str): Tên dataset để kiểm tra
        
    Returns:
        bool: True nếu dataset tồn tại, False nếu không
    """
    try:
        load_dotenv()
        client = Client()
        datasets = client.list_datasets()
        
        for dataset in datasets:
            if dataset.name.upper() == dataset_name.upper():
                return True
        return False
    except Exception as e:
        print(f"⚠️  Không thể kiểm tra dataset trên LangSmith: {e}")
        return False


def testing_dataset(method: str, dataset: str):
    """
    Kiểm tra dataset với phương pháp được chỉ định.
    Trước tiên sẽ kiểm tra xem dataset đã được tạo trên LangSmith chưa.
    """
    # Danh sách các phương pháp và dataset có sẵn
    available_methods = ['Zero_shot', 'PoT', 'CoT', 'PaL', 'MultiAgent']
    available_datasets = ['GSM8K', 'TATQA', 'TABMWP']
    
    # Xác định danh sách dataset cần kiểm tra
    datasets_to_test = []
    if dataset.lower() == 'all':
        datasets_to_test = available_datasets
    else:
        datasets_to_test = [dataset]
    
    # Kiểm tra xem các dataset đã tồn tại trên LangSmith chưa
    for ds in datasets_to_test:
        if not check_dataset_exists(ds):
            print(f"❌ Dataset {ds} chưa được tạo trên LangSmith!")
            print(f"📝 Vui lòng tạo dataset trước bằng lệnh:")
            print(f"   python mathqa.py create-dataset --dataset {ds}")
            print(f"   hoặc")
            print(f"   python mathqa.py create-dataset --dataset all --limit <số_lượng>")
            return
    
    # Xác định danh sách phương pháp cần kiểm tra
    methods_to_test = []
    if method.lower() == 'all':
        methods_to_test = available_methods
    else:
        methods_to_test = [method]
    
    print(f"✅ Tất cả dataset đã sẵn sàng trên LangSmith")
    print(f"🚀 Bắt đầu kiểm tra...")
    print(f"📊 Phương pháp: {', '.join(methods_to_test)}")
    print(f"📚 Dataset: {', '.join(datasets_to_test)}")
    print("-" * 60)
    
    # Chạy kiểm tra cho tất cả các kết hợp method-dataset
    total_tests = len(methods_to_test) * len(datasets_to_test)
    current_test = 0
    
    for test_method in methods_to_test:
        for test_dataset in datasets_to_test:
            current_test += 1
            print(f"\n🔄 Đang chạy kiểm tra {current_test}/{total_tests}: {test_method} trên {test_dataset}")
            print("=" * 50)
            
            try:
                test_case = TestingDataset(method=test_method, dataset=test_dataset)
                test_case.run()
                print(f"✅ Hoàn thành kiểm tra {test_method} trên {test_dataset}")
            except Exception as e:
                print(f"❌ Lỗi khi chạy {test_method} trên {test_dataset}: {str(e)}")
            
            print("=" * 50)
    
    print(f"\n🎉 Đã hoàn thành tất cả {total_tests} kiểm tra!")
    print(f"📈 Kết quả có thể xem trên LangSmith dashboard.")


def create_dataset_to_langsmith(dataset: str, limit: int):
    """
    Tạo dataset trên LangSmith với số lượng sample được chỉ định.
    
    Args:
        dataset (str): Tên dataset ('GSM8K', 'TATQA', 'TABMWP', hoặc 'all')
        limit (int): Số lượng sample để tạo
    """
    if dataset.upper() == "ALL":
        print(f"Đang tạo tất cả các dataset với {limit} samples...")
    else:
        print(f"Đang tạo dataset {dataset} với {limit} samples...")
    
    dataset_creator = DatasetToLangsmith(limit=limit)
    
    if dataset.upper() == "GSM8K":
        result = dataset_creator.create_gsm8k_dataset()
        print(f"✅ Đã tạo thành công dataset GSM8K với ID: {result.id}")
        
    elif dataset.upper() == "TATQA":
        result = dataset_creator.create_tatqa_dataset()
        print(f"✅ Đã tạo thành công dataset TATQA với ID: {result.id}")
        
    elif dataset.upper() == "TABMWP":
        result = dataset_creator.create_tabmwp_dataset()
        print(f"✅ Đã tạo thành công dataset TABMWP với ID: {result.id}")
        
    elif dataset.upper() == "ALL":
        results = dataset_creator.create_all_datasets()
        print("✅ Đã tạo thành công tất cả datasets:")
        for name, ds in results.items():
            print(f"  - {name.upper()}: {ds.id}")
    else:
        raise ValueError(f"Dataset không hợp lệ: {dataset}. Sử dụng 'GSM8K', 'TATQA', 'TABMWP', hoặc 'all'")
  
def main():
    parser = argparse.ArgumentParser(description="MathQA")
    print('-' * 50)
    subparsers = parser.add_subparsers(dest='command', title='Danh sách lệnh được hỗ trợ', description="Chọn một trong các lệnh bên dưới để sử dụng")

    # Single question
    single_parser = subparsers.add_parser('solve', help='Giải quyết một câu hỏi')
    single_parser.add_argument("--question", type=str, required=True, help="Câu hỏi cần giải quyết")
    single_parser.add_argument("--context", help="Ngữ cảnh tùy chọn để sử dụng", default="")
    single_parser.add_argument("--show-reasoning", action='store_true', help="Hiển thị mã nguồn đã được sinh ra", default=False)

    # Dataset testing
    test_parser = subparsers.add_parser('test', help='Kiểm tra một tập dữ liệu trên LangSmith')
    test_parser.add_argument("--method", type=str, required=True, help="Phương thức sử dụng", choices=['Zero_shot', 'PoT', 'CoT', 'PaL', 'MultiAgent', 'all'])
    test_parser.add_argument("--dataset", type=str, required=True, help="Tập dữ liệu để kiểm tra", choices=['GSM8K', 'TATQA', 'TABMWP', 'all'])

    # Create dataset on LangSmith
    langsmith_parser = subparsers.add_parser('create-dataset', help='Tạo dataset trên LangSmith')
    langsmith_parser.add_argument("--dataset", type=str, required=True, help="Tập dữ liệu để tạo", choices=['GSM8K', 'TATQA', 'TABMWP', 'all'])
    langsmith_parser.add_argument("--limit", type=int, default=300, help="Số lượng mẫu để tạo (mặc định: 300)")

    # Show datasets
    list_parser = subparsers.add_parser('datasets', help='Hiển thị danh sách các tập dữ liệu có sẵn')

    args = parser.parse_args()

    try:
        if args.command == 'solve':
            single_question(args.question, args.context, args.show_reasoning)

        elif args.command == 'test':
            testing_dataset(args.method, args.dataset)

        elif args.command == 'create-dataset':
            create_dataset_to_langsmith(args.dataset, args.limit)

        elif args.command == 'datasets':
            print("Available datasets:")
            print("-" * 50)
            print("GSM8K - bao gồm 8,500 bài toán toán học chất lượng cao cho học sinh tiểu học do các nhà văn vấn đề con người tạo ra.\n")
            print("TATQA - chứa 16,552 câu hỏi liên quan đến 2,757 ngữ cảnh kết hợp từ các báo cáo tài chính thực tế.\n")
            print("TABMWP - chứa 38,431 bài toán mở cấp độ yêu cầu lý luận toán học trên cả dữ liệu văn bản và bảng.\n")

        elif args.command == 'chart':
            fetcher = ExperimentDataFetcher(args.experiment_id)
            fetcher.fetch_data()

            chart_generator = ChartGenerate(args.methods, args.datasets, args.metric)
            chart_generator.generate_chart()
        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\nNgười dùng đã ngắt quá trình")
        sys.exit(1)
    except Exception as e:
        print(f"\nLỗi: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()