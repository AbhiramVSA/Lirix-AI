import logging
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from dataclasses import dataclass
from tavily import AsyncTavilyClient
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataclass for search parameters
@dataclass
class SearchDataclass:
    max_results: int

# Dataclass for research dependencies
@dataclass
class ResearchDependencies:
    search_deps: SearchDataclass

# Class to encapsulate the search agent and its functionality
class SearchAgentWrapper:
    def __init__(self, groq_api_key: str, tavily_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        # Initialize the Groq model
        self.model = GroqModel(
            model_name=model_name,
            api_key=groq_api_key,
        )

        # Initialize Tavily client
        self.tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

        # Initialize the search agent
        self.search_agent = Agent(
            self.model,
            deps_type=ResearchDependencies,
            system_prompt=(
                "You are a researcher that MUST use search tools to find information. "
                "You are not allowed to make up information or rely on your training data.\n\n"
                "When given a query, you MUST use the get_search tool to find information and return a paragraph summarizing the results.\n\n"
                "TOOL USE INSTRUCTIONS:\n"
                "1. Call get_search(query=\"your search query\") to gather information.\n"
                "2. Summarize the search results into a single paragraph.\n"
                "3. DO NOT proceed without using the search tool."
                "4. ALWAYS use the get_Search tool to find information on ANY query given to you.\n"
            ),
        )

        # Search tool for Tavily
        @self.search_agent.tool
        async def get_search(search_data: RunContext[ResearchDependencies], query: str) -> str:
            max_results = search_data.deps.search_deps.max_results
            try:
                logger.info(f"EXECUTING SEARCH QUERY: '{query}'")
                results = await self.tavily_client.get_search_context(query=query, max_results=max_results)
                logger.info(f"Received {len(results)} search results for query '{query}'")
                
                # Combine the search results into a single string
                combined_results = " ".join(results)
                return combined_results
            except Exception as e:
                logger.error(f"Error in get_search: {e}")
                raise

    async def do_search(self, query: str, max_results: int = 5) -> str:
        search_deps = SearchDataclass(max_results=max_results)
        deps = ResearchDependencies(search_deps=search_deps)
        try:
            logger.info(f"Starting search for query: '{query}'")
            result = await self.search_agent.run(query, deps=deps)
            logger.info("Search completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in do_search: {e}")
            raise