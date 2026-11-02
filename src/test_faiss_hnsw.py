
import numpy as np
import faiss

from embeddings.embedding_model import (
    generate_embedding,
    generate_embeddings
)

from vectorstore.faiss_store import (
    create_hnsw_index,
    create_ip_index
)


# STEP 1: Create sample documents

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


# STEP 4: Create HNSW index

hnsw_index = create_hnsw_index(
    embeddings,
    M=32,
    ef_construction=200,
    ef_search=32
)

print("\nHNSW INDEX CREATED")
print("Total vectors:", hnsw_index.ntotal)
print("Graph connections M:", 32)


# STEP 5: Create query

query = "How does FAISS find similar vectors?"

query_embedding = generate_embedding(query)

query_vector = np.asarray(
    [query_embedding],
    dtype=np.float32
)

faiss.normalize_L2(query_vector)


# STEP 6: Exact Flat search

flat_scores, flat_indices = flat_index.search(
    query_vector,
    5
)

exact_set = set(flat_indices[0].tolist())


# STEP 7: Test different efSearch values

for ef_search in [8, 16, 32, 64]:

    hnsw_index.hnsw.efSearch = ef_search

    scores, indices = hnsw_index.search(
        query_vector,
        5
    )

    approximate_set = set(indices[0].tolist())

    recall = len(
        exact_set.intersection(approximate_set)
    ) / 5

    print("\n" + "=" * 50)

    print("HNSW SEARCH")
    print("efSearch:", ef_search)
    print("Recall@5:", recall)

    for rank, idx in enumerate(indices[0], start=1):

        print(
            rank,
            documents[int(idx)]
        )
