from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic import BaseModel, Field
from load_api_key import Settings
from typing import Any
from dataclasses import dataclass
import nest_asyncio
import asyncio

settings = Settings()

@dataclass
class Result:
    print("")

@dataclass
class Dependencies:
    print("")

class AgentWrapper:
    def __init__(self, groq_api_key: str, model_name: str):
        self.model = GroqModel(
            model_name = model_name,
            api_key = groq_api_key,
        )

    self.agent = Agent(
        self.model,
        
    )