
import numpy as np

from embeddings.embedding_model import generate_embeddings
from embeddings.similarity import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    normalize_vector
)


sentences = [
    "I love learning artificial intelligence.",
    "I enjoy studying AI and machine learning.",
    "I like eating pizza."
]


# Step 1: Generate embeddings
vectors = generate_embeddings(sentences)


# Step 2: Normalize embeddings
normalized_vectors = np.array([
    normalize_vector(vector)
    for vector in vectors
])


# Step 3: Compare sentence 1 with sentence 2
vector_a = vectors[0]
vector_b = vectors[1]

print("\nCOMPARISON: AI SENTENCES")

print(
    "Cosine similarity:",
    cosine_similarity(vector_a, vector_b)
)

print(
    "Dot product:",
    dot_product(vector_a, vector_b)
)

print(
    "Euclidean distance:",
    euclidean_distance(vector_a, vector_b)
)


# Step 4: Compare sentence 1 with sentence 3
vector_c = vectors[2]

print("\nCOMPARISON: AI VS PIZZA")

print(
    "Cosine similarity:",
    cosine_similarity(vector_a, vector_c)
)

print(
    "Dot product:",
    dot_product(vector_a, vector_c)
)

print(
    "Euclidean distance:",
    euclidean_distance(vector_a, vector_c)
)


# Step 5: Verify normalized dot product
print("\nNORMALIZED VECTOR COMPARISON")

print(
    "Normalized dot product:",
    dot_product(normalized_vectors[0], normalized_vectors[1])
)

print(
    "Cosine similarity:",
    cosine_similarity(vector_a, vector_b)
)
