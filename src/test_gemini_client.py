from llm.gemini_client import generate_response

prompt = """
Explain embeddings in simple terms.
Give one real-world example.
"""

answer = generate_response(prompt)

print("\nGEMINI ANSWER:\n")
print(answer)