import os
from dotenv import load_dotenv
from google import genai

# Load local .env settings when present; deployment environments can supply
# the same values directly as process environment variables.
load_dotenv()

# Use the configured model unless a caller explicitly requests another one.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Cache the default client to reuse its connection/configuration between requests.
_cached_client = None


def get_client(api_key=None):
    """Get or initialize the Gemini client with an API key."""
    global _cached_client
    # An explicit key is isolated to this call; otherwise use/cache the env key.
    key = api_key or os.getenv("GEMINI_API_KEY")

    if not key:
        raise ValueError("GEMINI_API_KEY not found. Please configure it in .env or provide it in the UI.")

    # Reuse only the client created for the default environment credential.
    if _cached_client is not None and not api_key:
        return _cached_client

    # Construct the SDK client only after verifying that a key is available.
    client = genai.Client(api_key=key)
    if not api_key:
        _cached_client = client
    return client


def generate_response(prompt, api_key=None, model_name=None):
    """
    Send a prompt to Gemini and return its generated response.
    """
    # Reject blank prompts to avoid wasting a remote model request.
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    # Resolve credentials and model independently so callers can override either.
    client = get_client(api_key=api_key)
    model = model_name or MODEL_NAME

    # Send the complete prompt to Gemini and wait for its generated response.
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    # Treat a missing response body as a failed generation, not a valid answer.
    if not response.text:
        raise RuntimeError("Gemini returned an empty response")

    return response.text
