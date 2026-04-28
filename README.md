# Reflection Agent

A Twitter post generation agent that uses iterative reflection to improve tweet quality.

## Description

This project implements a reflection-based AI agent using LangChain and LangGraph. The agent generates Twitter posts based on user requests and iteratively reflects on them, providing critiques and recommendations for improvement. The process continues up to 3(can be configured) reflections to refine the tweet.

The agent consists of two main components:
- **Generator**: Creates initial and revised Twitter posts
- **Reflector**: Critiques the generated posts and suggests improvements

## Features

- AI-powered Twitter post generation
- Iterative reflection for quality improvement
- Configurable reflection cycles (defaults to 3)
- Fast inference using Groq's LLM
- Modular chain-based architecture

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd reflection-agent
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   # Activate the virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   (optional)
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
   LANGSMITH_API_KEY=your_api_key
   LANGSMITH_PROJECT="<your_project>"
   ```

## Usage

Run the main script:
```bash
python main.py
```

When prompted, enter your request for a Twitter post. The agent will:
1. Generate an initial tweet
2. Reflect on it and provide critique
3. Generate improved versions based on the reflection
4. Continue this cycle up to 6 times

The final output will be the refined tweet along with the number of reflections performed.

## Configuration

- **Reflection Count**: Modify the `should_reflect` function in `main.py` to change the maximum number of reflections (default: 3)
- **Prompts**: Customize the generation and reflection prompts in `chains.py`
- **LLM Model**: Change the model in `llm.py` or via the `GROQ_MODEL` environment variable

## Project Structure

- `main.py`: Main script with the LangGraph workflow
- `chains.py`: Defines the generation and reflection chains
- `llm.py`: LLM configuration using ChatGroq
- `requirements.txt`: Python dependencies
- `pyproject.toml`: Project metadata

## Dependencies

- `langchain`: Framework for building LLM applications
- `langgraph`: Library for creating stateful, graph-based workflows
- `python-dotenv`: Environment variable management
- `langchain-groq`: Groq integration for LangChain

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.