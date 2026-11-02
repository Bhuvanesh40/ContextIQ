
import numpy as np
import faiss

from embeddings.embedding_model import (
    generate_embedding,
    generate_embeddings
)

from vectorstore.faiss_store import (
    create_ivf_index,
    create_ip_index
)


# STEP 1: Create a larger demonstration dataset

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

for topic in topics:

    for variation in range(30):

        concept = concepts[variation % len(concepts)]
        purpose = purposes[variation % len(purposes)]

        sentence = (
            f"{topic} uses {concept} to {purpose}. "
            f"This is example number {variation + 1}."
        )

        documents.append(sentence)


print("TOTAL DOCUMENTS:", len(documents))


# STEP 2: Generate embeddings

embeddings = generate_embeddings(documents)

print("EMBEDDING SHAPE:", embeddings.shape)


# STEP 3: Create exact Flat index

flat_index = create_ip_index(embeddings)

print("\nFLAT INDEX CREATED")
print("Vectors:", flat_index.ntotal)


# STEP 4: Create IVF index

ivf_index = create_ivf_index(
    embeddings,
    nlist=4
)

print("\nIVF INDEX CREATED")
print("Vectors:", ivf_index.ntotal)
print("Number of clusters:", ivf_index.nlist)


# STEP 5: Create query

query = "How does FAISS find similar vectors?"

query_embedding = generate_embedding(query)

query_vector = np.asarray(
    [query_embedding],
    dtype=np.float32
)

faiss.normalize_L2(query_vector)


# STEP 6: Search exact Flat index

flat_scores, flat_indices = flat_index.search(
    query_vector,
    5
)


print("\nEXACT FLAT SEARCH RESULTS")

for rank, idx in enumerate(flat_indices[0], start=1):

    print(
        rank,
        documents[int(idx)]
    )


# STEP 7: Compare IVF with different nprobe values

for nprobe in [1, 2, 4]:

    ivf_index.nprobe = nprobe

    scores, indices = ivf_index.search(
        query_vector,
        5
    )

    # Calculate recall@5
    exact_set = set(flat_indices[0].tolist())
    approximate_set = set(indices[0].tolist())

    recall = len(
        exact_set.intersection(approximate_set)
    ) / 5

    print("\n" + "=" * 55)

    print("IVF SEARCH")
    print("NPROBE:", nprobe)

    print("RECALL@5:", recall)

    for rank, idx in enumerate(indices[0], start=1):

        print(
            rank,
            documents[int(idx)]
        )
