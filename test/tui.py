import asyncio
import nest_asyncio
import sys

sys.path.insert(1, 'C:/Users/abhir/Desktop/Lirix-AI/src/lirix')

from searchAgent import SearchAgentWrapper
from load_api_key import Settings

# Apply nest_asyncio to avoid event loop issues
nest_asyncio.apply()

# Load API keys
settings = Settings()
groq_api_key = settings.GROQ_API_KEY
tavily_api_key = settings.TAVILY_API_KEY

# Initialize the search agent
search_agent_wrapper = SearchAgentWrapper(groq_api_key, tavily_api_key, model_name="deepseek-r1-distill-llama-70b")

async def chatbot():
    print("\nTerminal Chatbot - Type your query (or type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        try:
            result_data = await search_agent_wrapper.do_search(query, max_results=5)
            print("\nBot:", result_data.data, "\n")
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    asyncio.run(chatbot())
