import os
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Send a test request
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Explain Retrieval-Augmented Generation in 3 simple sentences."
)

# Display response
print("\nGEMINI RESPONSE:\n")
print(response.text)