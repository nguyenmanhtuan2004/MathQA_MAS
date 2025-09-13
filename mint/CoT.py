import os
import re
import langsmith as ls

from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

class Step(BaseModel):
    explanation: str
    output: str
class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

class ChainOfThoughtPrompt:
    def __init__(self, model_name: str = "", model_provider: str = "", temperature: float = 0.0):
        load_dotenv()
        self.model_name = os.getenv("MODEL_NAME")
        self.model_provider = os.getenv("MODEL_PROVIDER")
        self.temperature = os.getenv("TEMPERATURE")
        self.model = init_chat_model(self.model_name, model_provider=self.model_provider, temperature=self.temperature)
        self.model_with_tools = self.model.with_structured_output(MathReasoning)

    def solve(self, question: str):
        user_content = f"\n\n# Question: {question}"

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
        ai_msg = self.model_with_tools.invoke(messages)
        predicted_answer = ai_msg.final_answer

        return predicted_answer, ai_msg.steps