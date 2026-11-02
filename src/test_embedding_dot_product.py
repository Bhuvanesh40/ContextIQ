
from embeddings.embedding_model import generate_embeddings
from embeddings.similarity import dot_product

sentences = [
    "I love learning artificial intelligence.",
    "I enjoy studying AI and machine learning.",
    "I like eating pizza."
]

# Generate embeddings
vectors = generate_embeddings(sentences)

# Calculate dot products
score_1 = dot_product(vectors[0], vectors[1])

score_2 = dot_product(vectors[0], vectors[2])

print("\nDOT PRODUCT RESULTS")

print("AI-related sentences:", score_1)

print("AI vs Pizza:", score_2)
