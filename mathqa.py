from typing import Literal, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from mint.PoT import ProgramOfThoughtsPrompt
from langgraph.checkpoint.memory import MemorySaver

import random
import argparse
import re
import sys
import os
import uuid


class State(TypedDict):
    question: str
    answer: Optional[str]
    context: Optional[str]
    error: Optional[str]
    debug_count: int  
    show_reasoning: bool

class IntermediateProgram(BaseModel):
    program: str

def PreProcessing(state: State):
    state["question"] = re.sub(r'\s+', ' ', state["question"].strip())                      
    print("Question: " + state["question"] + "\n")
    return {**state}

def CodeGenerator(state: State, cot: ProgramOfThoughtsPrompt):
    generated_code = cot.solve(state["question"], state["context"], state["show_reasoning"])
    return {**state, "debug_count": state.get('debug_count', 0), "answer": generated_code}

def Verifier(state: State, cot: ProgramOfThoughtsPrompt):
    error = cot.check_syntax_and_logic(state["answer"])
    return {**state, "debug_count": state.get('debug_count', 0), "error": error}

def Executor (state: State, cot: ProgramOfThoughtsPrompt):
    result, success = cot.safe_execute(str(state["answer"]))
    if success:
        return {**state, "debug_count": state.get('debug_count', 0), "answer": result}
    else:
        return {**state, "debug_count": state.get('debug_count', 0), "error": result}

def Debug_Feedback(state: State, cot: ProgramOfThoughtsPrompt):
    fixed_code = cot.fix_error(state["answer"], state["error"])
    return {**state, "debug_count": state.get('debug_count', 0) + 1, "error": None, "answer": fixed_code}

def Answer(state: State):
    answer = state.get("answer")
    if answer is not None:
        print("Answer: " + str(answer) + "\n")
    else:
        print("Answer: None\n")
    return {**state, "debug_count": state.get('debug_count', 0)}

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


def testing_dataset(method: str, dataset: str, limit: int):
    if method.lower() == "zero_shot":
        pass
    elif method.lower() == "pot":
        pass
    elif method.lower() == "cot":
        pass
    elif method.lower() == "pal":
        pass
    else:
        raise ValueError(f"Unknown method: {method}. Use 'Zero_shot', 'PoT', 'CoT', 'PaL'")
  
def main():
    print("MathQA")
    print('-' * 50)
    parser = argparse.ArgumentParser(description="MathQA")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Single question
    single_parser = subparsers.add_parser('solve', help='Solve a single question')
    single_parser.add_argument("--question", type=str, required=True, help="Question to solve")
    single_parser.add_argument("--context", help="Optional context to use", default="")
    single_parser.add_argument("--show-reasoning", action='store_true', help="Show the generated code", default=False)

    # Dataset testing
    dataset_parser = subparsers.add_parser('test', help='Test a dataset')
    dataset_parser.add_argument("--dataset", type=str, required=True, help="Dataset to test", choices=['GSM8K', 'TATQA', 'TABMWP'])
    dataset_parser.add_argument("--method", type=str, required=True, help="Prompting method to test", choices=['Zero_shot', 'PoT', 'CoT', 'PaL'])
    dataset_parser.add_argument("--num_samples", type=int,required=True, help="Number of samples to test")
    
    # Show datasets
    list_parser = subparsers.add_parser('datasets', help='Show available datasets')

    
    args = parser.parse_args()

    cot = ProgramOfThoughtsPrompt()

    builder = StateGraph(State)
    builder.add_node("PreProcessing", PreProcessing)
    builder.add_node("CodeGenerator", lambda state: CodeGenerator(state, cot))
    builder.add_node("Verifier", lambda state: Verifier(state, cot))
    builder.add_node("Executor", lambda state: Executor(state, cot))
    builder.add_node("Debug_Feedback", lambda state: Debug_Feedback(state, cot))
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

    try:
        if args.command == 'solve':
            state = State(
                question=args.question,
                context=args.context,
                answer=None,
                error=None,
                debug_count=0,
                show_reasoning=args.show_reasoning  
            )

            result = graph.invoke(
                input=state,
                config={"configurable": {"thread_id": str(uuid.uuid4())}}
            )

        elif args.command == 'test':
            pass

        elif args.command == 'datasets':
            print("Available datasets:")
            print("-" * 50)
            print("GSM8K - consists of 8.5K high quality grade school math problems created by human problem writers.\n")
            print("TATQA - contains 16,552 questions associated with 2,757 hybrid contexts from real-world financial reports.\n")
            print("TABMWP - contains 38,431 open-domain grade-level problems that require mathematical reasoning on both textual and tabular data.\n")

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()