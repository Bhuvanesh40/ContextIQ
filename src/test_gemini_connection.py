import os
from dotenv import load_dotenv

# Load environment variables from the project root
load_dotenv()

# Read the API key
api_key = os.getenv("GEMINI_API_KEY")

# Verify without displaying the actual key
if api_key:
    print("Gemini API key loaded successfully!")
else:
    print("Gemini API key not found!")