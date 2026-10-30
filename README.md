<p align="center">
  <img src="assets/lirix-mark.svg" alt="Lirix connected-node mark" width="60">
</p>

<h1 align="center">Lirix</h1>

<p align="center">Small Python experiments with language-model agents, search, and database queries.</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#source-modules">Source modules</a> ·
  <a href="#run-the-dice-example">Run an example</a> ·
  <a href="#limits">Limits</a> ·
  <a href="#license">License</a>
</p>

## Overview

This repository contains separate Python scripts. It does not contain an installable package or a dependency manifest with fixed versions.

The scripts use `pydantic_ai`, `pydantic_settings`, `dotenv`, `aiohttp`, `asyncpg`, and `tavily`. Install compatible packages for the script that you use. The repository does not define a tested install command.

## Source modules

| File | What it contains |
| --- | --- |
| `src/lirix/DiceAgent.py` | A Groq-backed chat loop with a tool that returns a random value from one to six. |
| `src/lirix/searchAgent.py` | A Groq-backed search agent that calls Tavily. |
| `src/lirix/Wiki_Agent.py` | Wikipedia search and summary code with a local JSON cache. |
| `src/lirix/SQL_Agent.py` | A Groq-backed query generator that sends SQL to PostgreSQL through `asyncpg`. |
| `src/lirix/agent_class.py` | An incomplete wrapper scaffold. |
| `src/lirix/load_api_key.py` | Settings for API keys and optional database values. |

## Run the dice example

Create a Python environment and install compatible dependencies for `DiceAgent.py`. The repository does not pin package versions.

Set `GROQ_API_KEY` and `TAVILY_API_KEY` in the environment or in a root `.env` file. `Settings` requires both values when this script starts.

From the repository root, run:

```bash
python src/lirix/DiceAgent.py
```

The script starts an interactive prompt. Enter `exit` to stop it. Runtime behavior was not tested for this README update.

## Database example

`SQL_Agent.py` also requires `SUPABASE_URL` and `SUPABASE_PASSWORD`. The source sends model-generated SQL to the database. It asks the model for `SELECT` statements, but it does not check the statement before execution. Use a database account with read-only access.

## Limits

The repository has no `pyproject.toml`, `setup.py`, `requirements.txt`, or package version metadata. The `test/` directory contains scripts, but the repository does not define a test command.

`agent_class.py` is not a usable general agent API. Treat each source file as a separate example. There is no shared package interface.

## Documentation

The project also lists the [Lirix documentation site](https://lirix-ai-documentation.vercel.app/).

## License

The repository includes a [CC0 1.0 Universal](LICENSE) dedication.
