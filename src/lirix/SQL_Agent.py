import asyncpg # type: ignore
import asyncio
from pydantic_ai.models.groq import GroqModel
from pydantic import BaseModel, Field
from annotated_types import MinLen
from pydantic_ai import Agent
from dataclasses import dataclass
from typing import Optional, Tuple, Annotated
from load_api_key import Settings
import os

# Load environment variables
settings = Settings()

# Database connection settings
SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_PASSWORD = settings.SUPABASE_PASSWORD

# Groq API Key
GROQ_API_KEY = settings.GROQ_API_KEY

# Database schema
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    title TEXT NOT NULL,
    content TEXT,
    published BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

# Pydantic models
class Success(BaseModel):
    type: str = Field("Success", pattern="^Success$")
    sql_query: Annotated[str, MinLen(1)]
    explanation: str

class InvalidRequest(BaseModel):
    type: str = Field("InvalidRequest", pattern="^InvalidRequest$")
    error_message: str

class Response(BaseModel):
    type: str

@dataclass
class Deps:
    conn: asyncpg.Connection
    db_schema: str = DB_SCHEMA

# Groq Model
model = GroqModel(
    model_name="llama-3.3-70b-versatile",
    api_key=settings.GROQ_API_KEY
)

# SQL Agent
sql_agent = Agent(
    model=model,
    deps_type=Deps,
    retries=3,
    result_type=Response,
    system_prompt=(
        f"""You are a proficient Database Administrator having expertise in generating SQL queries. Your task is to convert natural language requests into SQL queries for a PostgreSQL database.
        You must respond with a Success object containing a sql_query and an explanation.

        Database schema:
        {DB_SCHEMA}

        Format your response exactly like this, with no additional text or formatting:
        {{
            "type": "Success",
            "sql_query": "<your SQL query here>",
            "explanation": "<your explanation here>"
        }}

        If you cannot generate a valid query, respond with:
        {{
            "type": "InvalidRequest",
            "error_message": "<explanation of why the request cannot be processed>"
        }}

        Important:
        1. Respond with ONLY the JSON object, no additional text
        2. Always include the "type" field as either "Success" or "InvalidRequest"
        3. All queries must be SELECT statements
        4. Provide clear explanations
        5. Use proper JOIN conditions and WHERE clauses as needed
        """
    )
)

# Connect to the database
async def init_database() -> asyncpg.Connection:
    conn = await asyncpg.connect(SUPABASE_URL, password=SUPABASE_PASSWORD)
    return conn

# Execute SQL query
async def execute_query(conn: asyncpg.Connection, query: str) -> Tuple[bool, Optional[str]]:
    try:
        results = await conn.fetch(query)
        for record in results:
            print(dict(record))
        return True, None
    except Exception as e:
        return False, str(e)

# Query database through the agent
async def query_database(prompt: str, conn: asyncpg.Connection):
    try:
        result = await sql_agent.run(prompt, deps=Deps(conn=conn))
        if result.type == "Success":
            success, error = await execute_query(conn, result.sql_query)
            if not success:
                print(f"Query execution failed: {error}")
        else:
            print(f"Invalid request: {result.error_message}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Terminal chatbot loop
async def main():
    conn = await init_database()
    print("Connected to the database. Start asking your SQL questions!")

    try:
        while True:
            prompt = input("You: ")
            if prompt.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            await query_database(prompt, conn)
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
