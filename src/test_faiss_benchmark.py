
import time
import numpy as np
import faiss

from embeddings.embedding_model import generate_embeddings

from vectorstore.faiss_store import (
    create_ip_index,
    create_ivf_index,
    create_hnsw_index,
    create_ivfpq_index
)


# STEP 1: Create a demonstration dataset

topics = [
    "FAISS vector search",
    "RAG systems",
    "Text embeddings",
    "Python programming",
    "Database management",
    "Cloud computing",
    "Neural networks",
    "Information retrieval"
]

concepts = [
    "similarity search",
    "numerical representations",
    "data processing",
    "machine learning",
    "information storage"
]

purposes = [
    "improve search efficiency",
    "retrieve relevant information",
    "process large collections of data",
    "support intelligent applications",
    "organize information effectively",
    "compare related information"
]

documents = []

for i in range(10000):

    topic = topics[i % len(topics)]
    concept = concepts[i % len(concepts)]
    purpose = purposes[i % len(purposes)]

    documents.append(
        f"{topic} uses {concept} to {purpose}. "
        f"Example number {i}."
    )


print("TOTAL DOCUMENTS:", len(documents))


# STEP 2: Generate embeddings in batches

embedding_batches = []

for start in range(0, len(documents), 500):

    batch = documents[start:start + 500]

    batch_embeddings = generate_embeddings(batch)

    embedding_batches.append(batch_embeddings)


embeddings = np.vstack(embedding_batches)

print("EMBEDDING SHAPE:", embeddings.shape)


# STEP 3: Prepare test queries

queries = [
    "How does FAISS search similar vectors?",
    "What are text embeddings?",
    "How does RAG retrieve information?",
    "Explain Python programming.",
    "What is database management?",
    "How does cloud computing work?",
    "Explain neural networks.",
    "What is information retrieval?",
    "How can vector search improve efficiency?",
    "How are numerical representations used?"
]


query_batches = generate_embeddings(queries)

query_vectors = np.asarray(
    query_batches,
    dtype=np.float32
).copy()

faiss.normalize_L2(query_vectors)


# STEP 4: Create the exact Flat index

start_time = time.perf_counter()

flat_index = create_ip_index(embeddings)

flat_build_time = time.perf_counter() - start_time


# STEP 5: Create IVF index

start_time = time.perf_counter()

ivf_index = create_ivf_index(
    embeddings,
    nlist=16
)

ivf_build_time = time.perf_counter() - start_time

ivf_index.nprobe = 4


# STEP 6: Create HNSW index

start_time = time.perf_counter()

hnsw_index = create_hnsw_index(
    embeddings,
    M=32,
    ef_construction=200,
    ef_search=32
)

hnsw_build_time = time.perf_counter() - start_time


# STEP 7: Create IVF-PQ index

start_time = time.perf_counter()

pq_index = create_ivfpq_index(
    embeddings,
    nlist=16,
    M=8,
    nbits=8
)

pq_build_time = time.perf_counter() - start_time

pq_index.nprobe = 16


# STEP 8: Store index configurations

indexes = {
    "FlatIP": (flat_index, flat_build_time),
    "IVFFlat": (ivf_index, ivf_build_time),
    "HNSW": (hnsw_index, hnsw_build_time),
    "IVFPQ": (pq_index, pq_build_time)
}


# STEP 9: Generate exact baseline results

exact_results = []

for query_vector in query_vectors:

    scores, indices = flat_index.search(
        query_vector.reshape(1, -1),
        5
    )

    exact_results.append(set(indices[0].tolist()))


# STEP 10: Benchmark each index

print("\n" + "=" * 65)
print("FAISS BENCHMARK RESULTS")
print("=" * 65)

for name, (index, build_time) in indexes.items():

    total_search_time = 0
    total_recall = 0

    for query_number, query_vector in enumerate(query_vectors):

        start_time = time.perf_counter()

        scores, indices = index.search(
            query_vector.reshape(1, -1),
            5
        )

        search_time = time.perf_counter() - start_time

        total_search_time += search_time

        retrieved_set = set(indices[0].tolist())

        recall = len(
            exact_results[query_number].intersection(
                retrieved_set
            )
        ) / 5

        total_recall += recall


    average_latency = (
        total_search_time / len(query_vectors)
    ) * 1000

    average_recall = (
        total_recall / len(query_vectors)
    )

    # Serialized index size in MB
    index_size_mb = (
        faiss.serialize_index(index).nbytes
        / (1024 * 1024)
    )

    print("\nINDEX:", name)

    print(
        "BUILD TIME:",
        round(build_time, 4),
        "seconds"
    )

    print(
        "AVERAGE SEARCH LATENCY:",
        round(average_latency, 4),
        "milliseconds"
    )

    print(
        "AVERAGE RECALL@5:",
        round(average_recall, 4)
    )

    print(
        "SERIALIZED INDEX SIZE:",
        round(index_size_mb, 4),
        "MB"
    )

print("\nBENCHMARK COMPLETED")
