import os
import gc
import numpy as np

# Limit native math-library threads to reduce memory use on small cloud instances.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Cache each backend independently so model initialization happens only once.
_fastembed_model = None
_sentence_transformer_model = None


def get_embedding_model():
    """
    Load the embedding model with ultra-low memory consumption.
    Prefers FastEmbed (ONNX Runtime, ~230MB RAM) over PyTorch (~580MB RAM).
    """
    global _fastembed_model, _sentence_transformer_model

    # Prefer the lower-memory ONNX backend when it is installed and initializes.
    try:
        if _fastembed_model is None:
            from fastembed import TextEmbedding
            _fastembed_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        return "fastembed", _fastembed_model
    except Exception:
        # FastEmbed is optional; fall through to the supported transformer backend.
        pass

    # Use SentenceTransformer as a compatibility fallback when FastEmbed is absent.
    if _sentence_transformer_model is None:
        try:
            # Keep PyTorch's worker count small to avoid excess CPU/RAM allocation.
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass
        from sentence_transformers import SentenceTransformer
        _sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    return "sentence_transformers", _sentence_transformer_model


def generate_embedding(text):
    """Generate a single 384-dimensional dense embedding vector."""
    # Reject empty text because it has no meaningful content to encode.
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty")

    engine, model = get_embedding_model()

    # FastEmbed yields vectors lazily; consume the single result for this input.
    if engine == "fastembed":
        generator = model.embed([text])
        embedding = next(iter(generator))
        return np.asarray(embedding, dtype=np.float32)
    else:
        # Request a NumPy result from SentenceTransformer for FAISS compatibility.
        embedding = model.encode(text, convert_to_numpy=True)
        return np.asarray(embedding, dtype=np.float32)


def generate_query_embedding(query):
    """Generate embedding vector for a search query."""
    # Query and document text must use the same model/vector space.
    return generate_embedding(query)


def generate_embeddings(texts, batch_size=16):
    """Generate embeddings for a batch of text chunks with active garbage collection."""
    # Preserve the expected vector width even for an empty batch.
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    engine, model = get_embedding_model()

    # Use the selected backend's batch API to encode all chunks efficiently.
    if engine == "fastembed":
        generator = model.embed(texts, batch_size=batch_size)
        embeddings = np.array(list(generator), dtype=np.float32)
    else:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

    # Release temporary Python objects from batch processing before returning.
    gc.collect()
    return embeddings
