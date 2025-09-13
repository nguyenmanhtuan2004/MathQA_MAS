import re
import os

from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

class MathReasoning(BaseModel):
    final_answer: str

class ZeroShotPrompt:
    def __init__(self, model_name: str = "", model_provider: str = "", temperature: float = 0.0):
        load_dotenv()
        self.model_name = os.getenv("MODEL_NAME")
        self.model_provider = os.getenv("MODEL_PROVIDER")
        self.temperature = os.getenv("TEMPERATURE")
        self.model = init_chat_model(self.model_name, model_provider=self.model_provider, temperature=self.temperature)
        self.model_with_tools = self.model.with_structured_output(MathReasoning)

    def solve(self, question: str):
        user_content = f"\n\n# Question: {question}"

        zeroshot_message = [
            SystemMessage(content="""
            You are a math expert.
                - Include a `final_answer` as a single number, no units or symbols.
                - If you cannot solve it, return a final_answer of "unknown".
                - When dealing with money, do not round to thousands unless explicitly stated.
            """),
            HumanMessage(content=user_content)
        ]
        ai_msg = self.model_with_tools.invoke(zeroshot_message)
        predicted_answer = ai_msg.final_answer
        
        return predicted_answer