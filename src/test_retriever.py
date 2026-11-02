
from embeddings.embedding_model import (
    generate_embedding,
    generate_embeddings
)

from vectorstore.faiss_store import create_faiss_index

from vectorstore.retriever import retrieve_documents


# Step 1: Create document records
documents = [
    {
        "chunk_id": "chunk_0001",
        "source": "sample.pdf",
        "page_number": 1,
        "chunk_text": "RAG combines information retrieval with large language models."
    },
    {
        "chunk_id": "chunk_0002",
        "source": "sample.pdf",
        "page_number": 2,
        "chunk_text": "Embeddings convert text into numerical vectors."
    },
    {
        "chunk_id": "chunk_0003",
        "source": "sample.pdf",
        "page_number": 3,
        "chunk_text": "FAISS is a library for efficient vector similarity search."
    },
    {
        "chunk_id": "chunk_0004",
        "source": "python_guide.pdf",
        "page_number": 5,
        "chunk_text": "Python is a popular programming language."
    },
    {
        "chunk_id": "chunk_0005",
        "source": "llm_guide.pdf",
        "page_number": 4,
        "chunk_text": "Large language models generate natural language responses."
    }
]


# Step 2: Extract text from document records
texts = [
    document["chunk_text"]
    for document in documents
]


# Step 3: Generate embeddings
embeddings = generate_embeddings(texts)


# Step 4: Create FAISS index
index = create_faiss_index(embeddings)


# Step 5: Create user query
query = "What is used for searching similar vectors?"


# Step 6: Generate query embedding
query_embedding = generate_embedding(query)


# Step 7: Retrieve documents with metadata
results = retrieve_documents(
    index=index,
    query_embedding=query_embedding,
    documents=documents,
    top_k=3
)


# Step 8: Display results
print("\nRETRIEVED DOCUMENTS")

for rank, result in enumerate(results, start=1):

    print("\n" + "=" * 55)

    print("RANK:", rank)

    print("CHUNK ID:", result["chunk_id"])

    print("SOURCE:", result["source"])

    print("PAGE:", result["page_number"])

    print(
        "SIMILARITY:",
        round(result["similarity_score"], 4)
    )

    print("CONTENT:", result["chunk_text"])
