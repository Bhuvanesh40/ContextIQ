import gc
import os

# Limit thread allocations to prevent memory ballooning on cloud containers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_model = None


def get_embedding_model():
    """Lazy load the embedding model to minimize idle memory usage."""
    global _model
    if _model is None:
        import torch
        torch.set_num_threads(1)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def generate_embedding(text):
    """Generate embedding vector for a single text chunk."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty")

    model = get_embedding_model()
    embedding = model.encode(
        text,
        convert_to_numpy=True
    )
    return embedding


def generate_query_embedding(query):
    """Generate embedding vector for a search query."""
    return generate_embedding(query)


def generate_embeddings(texts, batch_size=16):
    """
    Generate embedding vectors for a batch of text chunks using a memory-safe batch size.
    """
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    gc.collect()
    return embeddings
