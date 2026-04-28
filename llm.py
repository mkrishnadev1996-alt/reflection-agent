from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

llm = ChatGroq(
    model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),  # Default to a specific model if not set in .env
    api_key=os.environ.get("GROQ_API_KEY")
    )

if __name__ == "__main__":
    # Test the LLM with a simple prompt
    response = llm.invoke("Hello, how are you?")
    print(response)