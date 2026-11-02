
from embeddings.embedding_model import (
    generate_embedding,
    generate_embeddings
)

from vectorstore.faiss_store import (
    create_faiss_index,
    search_faiss
)


# Step 1: Sample document chunks
documents = [
    "RAG combines information retrieval with large language models.",
    "Embeddings convert text into numerical vectors.",
    "FAISS is a library for efficient vector similarity search.",
    "Python is a popular programming language.",
    "Large language models generate natural language responses."
]


# Step 2: Generate document embeddings
document_embeddings = generate_embeddings(documents)


# Step 3: Create FAISS index
index = create_faiss_index(document_embeddings)

print("\nFAISS INDEX CREATED")

print("Total vectors:", index.ntotal)
print("Vector dimensions:", index.d)


# Step 4: Create user query
query = "What is used to search similar vectors?"

print("\nUSER QUERY:", query)


# Step 5: Generate query embedding
query_embedding = generate_embedding(query)


# Step 6: Search FAISS
scores, indices = search_faiss(
    index,
    query_embedding,
    top_k=3
)


# Step 7: Display retrieved results
print("\nTOP 3 SEARCH RESULTS")

for rank, (score, idx) in enumerate(
    zip(scores, indices),
    start=1
):

    print("\n" + "=" * 50)
    print("RANK:", rank)
    print("DOCUMENT INDEX:", idx)
    print("SIMILARITY SCORE:", round(float(score), 4))
    print("TEXT:", documents[int(idx)])
