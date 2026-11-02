
import faiss
import numpy as np


def create_faiss_index(embeddings):

    # Step 1: Convert embeddings into float32
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # Step 2: Validate dimensions
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Embeddings must be a non-empty 2D array")

    # Step 3: Normalize vectors
    embeddings = embeddings.copy()
    faiss.normalize_L2(embeddings)

    # Step 4: Get embedding dimensions
    dimension = embeddings.shape[1]

    # Step 5: Create FAISS index
    index = faiss.IndexFlatIP(dimension)

    # Step 6: Add embeddings
    index.add(embeddings)

    return index


def search_faiss(index, query_embedding, top_k=3):

    # Convert query into float32
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    if query_embedding.shape[1] != index.d:
        raise ValueError("Query embedding dimension mismatch")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # Normalize query
    query_embedding = query_embedding.copy()
    faiss.normalize_L2(query_embedding)

    # Search the index
    k = min(top_k, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        k
    )

    return scores[0], indices[0]


def create_l2_index(embeddings):

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Embeddings must be a non-empty 2D array")

    dimension = embeddings.shape[1]

    # Create exact L2 index
    index = faiss.IndexFlatL2(dimension)

    # Add document vectors
    index.add(embeddings)

    return index


def create_ip_index(embeddings):

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Embeddings must be a non-empty 2D array")

    # Copy before normalization
    embeddings = embeddings.copy()

    # Normalize vectors
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # Create inner product index
    index = faiss.IndexFlatIP(dimension)

    # Add normalized document vectors
    index.add(embeddings)

    return index

def create_ivf_index(embeddings, nlist=4):

    # Step 1: Convert embeddings to float32
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Embeddings must be a non-empty 2D array")

    if not 1 <= nlist < len(embeddings):
        raise ValueError("nlist must be positive and smaller than the number of vectors")

    # Step 2: Copy and normalize embeddings
    embeddings = embeddings.copy()
    faiss.normalize_L2(embeddings)

    # Step 3: Get vector dimensions
    dimension = embeddings.shape[1]

    # Step 4: Create a quantizer
    quantizer = faiss.IndexFlatIP(dimension)

    # Step 5: Create IVF index
    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    # Step 6: Train the IVF index
    index.train(embeddings)

    # Step 7: Add document vectors
    index.add(embeddings)

    return index

def create_hnsw_index(
    embeddings,
    M=32,
    ef_construction=200,
    ef_search=32
):

    # Step 1: Convert embeddings to float32
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Embeddings must be a non-empty 2D array")

    # Step 2: Validate parameters
    if M <= 0:
        raise ValueError("M must be positive")

    if ef_construction < M:
        raise ValueError("ef_construction should be >= M")

    if ef_search <= 0:
        raise ValueError("ef_search must be positive")

    # Step 3: Copy and normalize embeddings
    embeddings = embeddings.copy()
    faiss.normalize_L2(embeddings)

    # Step 4: Get vector dimensions
    dimension = embeddings.shape[1]

    # Step 5: Create HNSW index
    index = faiss.IndexHNSWFlat(
        dimension,
        M,
        faiss.METRIC_INNER_PRODUCT
    )

    # Step 6: Configure graph construction
    index.hnsw.efConstruction = ef_construction

    # Step 7: Configure search
    index.hnsw.efSearch = ef_search

    # Step 8: Add vectors
    index.add(embeddings)

    return index

def create_ivfpq_index(
    embeddings,
    nlist=4,
    M=8,
    nbits=8
):

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Embeddings must be a non-empty 2D array")

    dimension = embeddings.shape[1]

    if not 1 <= nlist < len(embeddings):
        raise ValueError("Invalid nlist")

    if M <= 0 or dimension % M != 0:
        raise ValueError("M must divide the vector dimension")

    if not 1 <= nbits <= 8:
        raise ValueError("nbits must be between 1 and 8")

    if len(embeddings) < 39 * (2 ** nbits):
        raise ValueError(
            "Not enough training vectors for this PQ configuration. "
            "Use more training data or a smaller nbits value."
        )

    embeddings = embeddings.copy()
    faiss.normalize_L2(embeddings)

    # Create IVF quantizer
    quantizer = faiss.IndexFlatIP(dimension)

    # Create IVF-PQ index
    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        nlist,
        M,
        nbits,
        faiss.METRIC_INNER_PRODUCT
    )

    # Train the index
    index.train(embeddings)

    # Add compressed vectors
    index.add(embeddings)

    return index
