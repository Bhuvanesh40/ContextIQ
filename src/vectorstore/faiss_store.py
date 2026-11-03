
import faiss
import numpy as np


def create_faiss_index(embeddings):
    # Build an exact inner-product index for normalized cosine search.
    # FAISS expects contiguous numeric arrays in float32 for this index type.
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # Require at least one vector and a 2D (count, dimension) input matrix.
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Embeddings must be a non-empty 2D array")

    # Normalize a copy so inner product is equivalent to cosine similarity.
    embeddings = embeddings.copy()
    faiss.normalize_L2(embeddings)

    # Step 4: Get embedding dimensions
    dimension = embeddings.shape[1]

    # IndexFlatIP performs exact inner-product search without training.
    index = faiss.IndexFlatIP(dimension)

    # Step 6: Add embeddings
    index.add(embeddings)

    return index


def search_faiss(index, query_embedding, top_k=3):
    # Return scores and vector positions for the query's nearest neighbors.
    # FAISS searches a matrix, so reshape the single query into one row.
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    # Prevent comparisons between vectors from different embedding spaces.
    if query_embedding.shape[1] != index.d:
        raise ValueError("Query embedding dimension mismatch")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # Normalize the query exactly as the indexed vectors were normalized.
    query_embedding = query_embedding.copy()
    faiss.normalize_L2(query_embedding)

    # Asking for more vectors than the index contains is unnecessary.
    k = min(top_k, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        k
    )

    return scores[0], indices[0]


def create_l2_index(embeddings):
    # Create an exact Euclidean-distance index for vector experiments.
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Embeddings must be a non-empty 2D array")

    dimension = embeddings.shape[1]

    # IndexFlatL2 compares every stored vector and returns squared L2 distances.
    index = faiss.IndexFlatL2(dimension)

    # Add document vectors
    index.add(embeddings)

    return index


def create_ip_index(embeddings):
    # Create a normalized exact inner-product index.
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
    # Create an inverted-file index that searches selected vector clusters.
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

    # The quantizer assigns vectors to coarse clusters using exact inner product.
    quantizer = faiss.IndexFlatIP(dimension)

    # The IVF index stores vectors in nlist partitions for faster approximate search.
    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    # Training learns the cluster centroids needed to partition the vectors.
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
    # Create a graph-based approximate nearest-neighbor index.
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

    # HNSW links neighboring vectors in a graph for efficient approximate search.
    index = faiss.IndexHNSWFlat(
        dimension,
        M,
        faiss.METRIC_INNER_PRODUCT
    )

    # Construction effort controls graph quality and indexing cost.
    index.hnsw.efConstruction = ef_construction

    # Search effort controls the speed/recall trade-off at query time.
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
    # Create an IVF index with product-quantized, compressed vectors.
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

    # A full-precision quantizer chooses the coarse cluster for each vector.
    quantizer = faiss.IndexFlatIP(dimension)

    # Product quantization compresses sub-vectors to reduce index memory use.
    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        nlist,
        M,
        nbits,
        faiss.METRIC_INNER_PRODUCT
    )

    # Both IVF clustering and PQ codebooks must be learned before adding data.
    index.train(embeddings)

    # Add the training vectors after the index has learned its codebooks.
    index.add(embeddings)

    return index
