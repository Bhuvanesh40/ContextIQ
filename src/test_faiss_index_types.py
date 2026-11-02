
from embeddings.embedding_model import (
    generate_embedding,
    generate_embeddings
)

from vectorstore.faiss_store import (
    create_l2_index,
    create_ip_index
)


# Step 1: Create sample documents

documents = [
    "RAG combines information retrieval with large language models.",
    "Embeddings convert text into numerical vectors.",
    "FAISS is a library for efficient vector similarity search.",
    "Python is a popular programming language.",
    "Large language models generate natural language responses."
]


# Step 2: Generate embeddings

embeddings = generate_embeddings(documents)


# Step 3: Create both indexes

l2_index = create_l2_index(embeddings)

ip_index = create_ip_index(embeddings)


print("\nINDEX INFORMATION")

print("L2 index vectors:", l2_index.ntotal)

print("IP index vectors:", ip_index.ntotal)

print("Vector dimensions:", l2_index.d)


# Step 4: Create query

query = "What is used to search similar vectors?"

query_embedding = generate_embedding(query)

import numpy as np

query_vector = np.asarray(
    [query_embedding],
    dtype=np.float32
)


# Step 5: Search using L2

l2_distances, l2_indices = l2_index.search(
    query_vector,
    3
)


# Step 6: Search using IP

ip_query = query_vector.copy()

import faiss

faiss.normalize_L2(ip_query)

ip_scores, ip_indices = ip_index.search(
    ip_query,
    3
)


# Step 7: Display L2 results

print("\nINDEX FLAT L2 RESULTS")

for rank, (distance, idx) in enumerate(
    zip(l2_distances[0], l2_indices[0]),
    start=1
):

    print("\nRank:", rank)
    print("Distance:", round(float(distance), 4))
    print("Text:", documents[int(idx)])


# Step 8: Display IP results

print("\nINDEX FLAT IP RESULTS")

for rank, (score, idx) in enumerate(
    zip(ip_scores[0], ip_indices[0]),
    start=1
):

    print("\nRank:", rank)
    print("Similarity:", round(float(score), 4))
    print("Text:", documents[int(idx)])
