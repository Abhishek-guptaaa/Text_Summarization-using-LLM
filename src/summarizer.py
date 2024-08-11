from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables from .env file
load_dotenv()

# Initialize the Groq model with API key from environment
def initialize_model(model_name="mixtral-8x7b-32768", temperature=0.0):
    api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )

def create_prompt_template():
    """Create and return a prompt template for summarization."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that summarizes text for a 5-year-old. Make it very simple and easy to understand.",
            ),
            ("human", "{input}"),
        ]
    )

def generate_summary(model, prompt_template, text):
    """Generate a summary using the model and prompt template."""
    chain = prompt_template | model
    summary = chain.invoke({"input": text})
    return summary
