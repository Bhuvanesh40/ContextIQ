
import numpy as np

from embeddings.embedding_model import generate_embeddings
from embeddings.similarity import normalize_vector


sentences = [
    "I love learning artificial intelligence.",
    "I enjoy studying AI and machine learning.",
    "I like eating pizza."
]

# Generate embeddings
vectors = generate_embeddings(sentences)

print("\nBEFORE NORMALIZATION")

for index, vector in enumerate(vectors):

    print(
        f"Sentence {index + 1} magnitude:",
        np.linalg.norm(vector)
    )


# Normalize embeddings
normalized_vectors = np.array([
    normalize_vector(vector)
    for vector in vectors
])


print("\nAFTER NORMALIZATION")

for index, vector in enumerate(normalized_vectors):

    print(
        f"Sentence {index + 1} magnitude:",
        np.linalg.norm(vector)
    )


print("\nNORMALIZED EMBEDDING SHAPE:")

print(normalized_vectors.shape)
