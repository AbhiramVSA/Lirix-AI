import asyncio
import nest_asyncio
import json
import logging
import sys
import os
import traceback

sys.path.insert(1, 'C:/Users/abhir/Desktop/Lirix-AI/src/lirix')

from searchAgent import SearchAgentWrapper
from load_api_key import Settings

# Set up logging to see more details about what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

settings = Settings()
nest_asyncio.apply()

groq_api_key = settings.GROQ_API_KEY
tavily_api_key = settings.TAVILY_API_KEY

# Create the search agent wrapper
search_agent_wrapper = SearchAgentWrapper(groq_api_key, tavily_api_key, model_name="deepseek-r1-distill-llama-70b")

async def main():
    # Perform a search
    query = "land prices in paris in 2025"
    max_results = 5
    try:
        print(f"Executing search query: '{query}'")
        result_data = await search_agent_wrapper.do_search(query, max_results)

        # Print the search result (a single paragraph string)
        if result_data:
            print("\n==== SEARCH RESULTS ====")
            print(result_data.data)

        else:
            print("No result data found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        # Print the full traceback for better debugging
        traceback.print_exc()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())