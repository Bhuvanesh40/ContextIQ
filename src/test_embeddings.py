
from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embedding(text):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty")

    # Convert text into a numerical vector
    embedding = model.encode(
        text,
        convert_to_numpy=True
    )

    return embedding


def generate_embeddings(texts):

    if not texts:
        return []

    # Convert multiple texts into vectors
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings

from embeddings.embedding_model import (
    generate_embedding,
    generate_embeddings
)

sentences = [
    "I love learning artificial intelligence.",
    "I enjoy studying AI and machine learning.",
    "I like eating pizza."
]

# Generate one embedding
vector = generate_embedding(sentences[0])

print("\nSINGLE EMBEDDING")

print("Vector:", vector)

print("Dimensions:", len(vector))


# Generate embeddings for multiple sentences
vectors = generate_embeddings(sentences)

print("\nMULTIPLE EMBEDDINGS")

print("Shape:", vectors.shape)

print("\nFirst sentence vector:")

print(vectors[0])