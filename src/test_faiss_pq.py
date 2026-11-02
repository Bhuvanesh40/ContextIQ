
import numpy as np
import faiss

from embeddings.embedding_model import (
    generate_embedding,
    generate_embeddings
)

from vectorstore.faiss_store import (
    create_ip_index,
    create_ivfpq_index
)


# STEP 1: Generate a larger dataset

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


# STEP 3: Create exact Flat index

flat_index = create_ip_index(embeddings)

print("\nFLAT INDEX CREATED")


# STEP 4: Create IVF-PQ index

pq_index = create_ivfpq_index(
    embeddings,
    nlist=16,
    M=8,
    nbits=8
)

print("\nIVF-PQ INDEX CREATED")
print("Vectors:", pq_index.ntotal)
print("Clusters:", pq_index.nlist)
print("Subvectors:", pq_index.pq.M)
print("Bits per code:", pq_index.pq.nbits)


# STEP 5: Create query

query = "How does FAISS perform vector similarity search?"

query_embedding = generate_embedding(query)

query_vector = np.asarray(
    [query_embedding],
    dtype=np.float32
)

faiss.normalize_L2(query_vector)


# STEP 6: Exact search

flat_scores, flat_indices = flat_index.search(
    query_vector,
    5
)



# STEP 7: Test different nprobe values

exact_set = set(flat_indices[0].tolist())

for nprobe in [1, 2, 4, 8, 16]:

    pq_index.nprobe = nprobe

    pq_scores, pq_indices = pq_index.search(
        query_vector,
        5
    )

    # Calculate Recall@5
    pq_set = set(pq_indices[0].tolist())

    recall = len(
        exact_set.intersection(pq_set)
    ) / 5

    print("\n" + "=" * 50)

    print("IVF-PQ SEARCH")
    print("NPROBE:", nprobe)
    print("RECALL@5:", recall)

    for rank, idx in enumerate(
        pq_indices[0],
        start=1
    ):

        print(
            rank,
            documents[int(idx)]
        )
