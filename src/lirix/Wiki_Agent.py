from pydantic_ai import Agent, RunContext, models
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelRequest, UserPromptPart
from pydantic_ai.models.groq import GroqModel
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Optional
import asyncio
import aiohttp
import json
import os
import re

CACHE_FILE = "wiki_cache.json"

@dataclass
class SearchResult:
    title: str
    snippet: str

@dataclass
class WikiResults:
    results: List[SearchResult]

class WikiAgent:
    def __init__(self, model: BaseModel, wiki_api_key: str, system_prompt: str = (
        "You are a helpful assistant. You can have a conversation with the user and answer general questions. "
        "always use the Wikipedia search tool "
        "Make sure to summarize the content given out by the search_wikipedia tool. "
        "Otherwise, respond as a normal chatbot."
    )):
        self.model = model
        self.wiki_api_key = wiki_api_key
        self.cache = self.load_cache()
        
        @dataclass
        class Deps:
            pass
        
        self.deps = Deps()
        self.agent = Agent(model=self.model, system_prompt=system_prompt, deps_type=Deps, retries=2)
        self.messages: List[ModelMessage] = []
        self._register_tools()

    def _register_tools(self):
        @self.agent.tool_plain(retries=1)
        async def search_wikipedia(query: str) -> str:
            '''Searches the given query on Wikipedia'''
            return await self.instance_search_wikipedia(query)

    def load_cache(self):
        """Loads cache from the JSON file."""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as file:
                    return json.load(file)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def save_cache(self):
        """Saves updated cache to JSON file."""
        with open(CACHE_FILE, 'w') as file:
            json.dump(self.cache, file, indent=4)

    @staticmethod
    def clean_html_tags(text: str) -> str:
        """Removes HTML tags like <span> from the Wikipedia search snippet."""
        return re.sub(r'<.*?>', '', text) 

    async def fetch_wikipedia_summary(self, page_id: int) -> str:
        """Fetch a full summary of the Wikipedia article."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": "1",  # Get only the first section (summary)
            "explaintext": "0",  # Remove HTML
            "pageids": page_id
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data["query"]["pages"].get(str(page_id), {}).get("extract", "No additional information available.")

    async def instance_search_wikipedia(self, query: str) -> str:
        '''Instance method to search Wikipedia.'''
        if not isinstance(self.cache, dict):
            self.cache = {}

        if query.lower() in (cached_query.lower() for cached_query in self.cache):
            return f"🔍 Cached Wikipedia Search: {self.cache[query]}"
        
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "utf8": 1,
            "srlimit": 5  # Fetch more results
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data_text = await response.text()
                try:
                    data = json.loads(data_text)  # Convert string to JSON
                except json.JSONDecodeError:
                    print("Error: Failed to parse JSON. Response was:", data_text)
                    return "Error: Unable to retrieve Wikipedia data."
                
                search_results = data.get("query", {}).get("search", [])

                if not search_results:
                    return "No results found."
                
                result_text = []
                for result in search_results[:5]:  # Limit to 5 results
                    title = result.get('title', 'Unknown title')
                    snippet = result.get('snippet', 'No snippet available')
                    page_id = result.get("pageid", None)
                    
                    summary = await self.fetch_wikipedia_summary(page_id) if page_id else snippet
                    result_text.append(f"**{title}**: {self.clean_html_tags(summary)}...")
                
                full_result = "\n\n".join(result_text)
                self.cache[query] = full_result
                self.save_cache()
            return full_result  # Returning raw results to be summarized later

    async def process_user_input(self, user_input: str):
        """Processes user input and calls Wikipedia search if needed."""
        if user_input.lower() == "clear":
            self.messages.clear()
            return "Conversation cleared."

        if "search the wiki" in user_input.lower():
            query = user_input.replace("search the wiki", "").strip()
            search_result = await self.instance_search_wikipedia(query=query)

            # **Pass the search results to the LLM for summarization**
            llm_prompt = (
                "Here are some Wikipedia search results:\n\n"
                f"{search_result}\n\n"
                "Summarize the key information concisely."
            )

            response = await self.agent.run(
                user_prompt=llm_prompt, message_history=self.messages, deps=self.deps
            )

            return f"🔍 Wikipedia Summary:\n\n{response.data}"

        self.messages.append(ModelRequest(parts=[UserPromptPart(content=user_input)]))
        self.messages[:] = self.messages[-5:]  # Keep last 5 messages for context
        response = await self.agent.run(user_prompt=user_input, message_history=self.messages, deps=self.deps)

        return f"Bot: {response.data}"

# Usage Example
if __name__ == "__main__":
    def get_model(model_type: str, model_name: str, api_key: str) -> BaseModel:
        if model_type.lower() == "groq":
            return GroqModel(model_name=model_name, api_key=api_key)
        raise ValueError(f"Unsupported model type: {model_type}")
    
    