from preprocessing.text_cleaner import clean_text


sample_text = """
RAG     is a technique.


It combines   information retrieval
with AI models.


Embeddings   represent text numerically!!!
"""


cleaned_text = clean_text(sample_text)

print("ORIGINAL TEXT:")
print(sample_text)

print("\nCLEANED TEXT:")
print(cleaned_text)