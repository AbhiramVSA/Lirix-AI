from typing import List, Any, Dict
import asyncio
from dataclasses import dataclass
from random import randint

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelRequest, UserPromptPart, TextPart
from pydantic_ai.models.groq import GroqModel

from load_api_key import Settings

settings = Settings()


class DiceAgent:
    """A reusable class for a dice-rolling AI agent with natural language betting."""

    def __init__(self, llm: str, api_key: str, system_prompt: str = "You have access to a dice rolling tool, use it when required. Comply to requests from the user"):
        self.llm = llm
        self.api_key = api_key

        self.model = GroqModel(
            model_name=llm,
            api_key=api_key,
        )

        @dataclass
        class Deps:
            pass  # No external dependencies for now

        self.deps = Deps()

        self.agent = Agent(
            model=self.model,
            system_prompt=system_prompt,
            deps_type=Deps,
            retries=2,
        )

        # Track conversation messages
        self.messages: List[ModelMessage] = []

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register the necessary tools for the agent."""

        @self.agent.tool_plain(retries=1)
        async def rolldie() -> int:
            """Rolls a die and returns a number between 1 and 6."""
            print("Rolling a die!")
            return randint(1, 6)

    async def process_user_input(self, user_input: str):
        """Processes user input, extracts bets, and determines results."""

        if user_input.lower() == "clear":
            self.messages.clear()
            return "Conversation cleared."

        if user_input.lower() == "exit":
            return "Exiting..."

        self.messages.append(ModelRequest(parts=[UserPromptPart(content=user_input)]))
        self.messages[:] = self.messages[-5:]  # Keep only the last 5 messages

        # Run LLM with context
        response = await self.agent.run(user_prompt=user_input, message_history=self.messages, deps=self.deps)

        # Handle tool calls (like rolldie)
        if isinstance(response.data, str):
            return f"Bot: {response.data}"

        elif isinstance(response.data, list):  # Tool calls come as a list
            for tool_call in response.data:
                if tool_call["name"] == "rolldie":
                    roll_result = (await self.agent.run("rolldie")).data
                    return f"🎲 Dice rolled: {roll_result}"

        return "Bot: I'm processing that information."

# Usage Example
if __name__ == "__main__":
    async def main():
        dice_agent = DiceAgent(
            llm="llama-3.3-70b-versatile",
                api_key=settings.GROQ_API_KEY,
            )

        while True:
            user_message = input("You: ")
            response = await dice_agent.process_user_input(user_message)

            if response == "Exiting...":
                break

            print(response)

    asyncio.run(main())
