import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

_cached_client = None


def get_client(api_key=None):
    """Get or initialize the Gemini client with an API key."""
    global _cached_client
    key = api_key or os.getenv("GEMINI_API_KEY")

    if not key:
        raise ValueError("GEMINI_API_KEY not found. Please configure it in .env or provide it in the UI.")

    if _cached_client is not None and not api_key:
        return _cached_client

    client = genai.Client(api_key=key)
    if not api_key:
        _cached_client = client
    return client


def generate_response(prompt, api_key=None, model_name=None):
    """
    Send a prompt to Gemini and return its generated response.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    client = get_client(api_key=api_key)
    model = model_name or MODEL_NAME

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response")

    return response.text
