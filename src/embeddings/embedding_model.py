from sentence_transformers import SentenceTransformer

# Load the embedding model (384-dimensional dense vectors)
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embedding(text):
    """Generate embedding vector for a single text chunk."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty")

    return model.encode(
        text,
        convert_to_numpy=True
    )


def generate_query_embedding(query):
    """Generate embedding vector for a search query."""
    return generate_embedding(query)


def generate_embeddings(texts):
    """Generate embedding vectors for a batch of text chunks."""
    if not texts:
        return []

    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )
