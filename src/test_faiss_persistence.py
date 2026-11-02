
from embeddings.embedding_model import (
    generate_embedding,
    generate_embeddings
)

from vectorstore.faiss_store import create_faiss_index

from vectorstore.persistence import (
    save_vectorstore,
    load_vectorstore
)

from vectorstore.retriever import retrieve_documents


# STEP 1: Create sample document records

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


# STEP 2: Generate document embeddings

texts = [
    document["chunk_text"]
    for document in documents
]

embeddings = generate_embeddings(texts)


# STEP 3: Create FAISS index

index = create_faiss_index(embeddings)

print("\nORIGINAL INDEX")
print("Vectors:", index.ntotal)


# STEP 4: Save index and metadata

save_vectorstore(
    index=index,
    documents=documents,
    index_path="data/index/faiss.index",
    metadata_path="data/index/metadata.json"
)


# STEP 5: Load saved index and metadata

loaded_index, loaded_documents = load_vectorstore(
    index_path="data/index/faiss.index",
    metadata_path="data/index/metadata.json"
)


print("\nLOADED INDEX")
print("Vectors:", loaded_index.ntotal)

print("Metadata records:", len(loaded_documents))


# STEP 6: Search the loaded index

query = "What is used for searching similar vectors?"

query_embedding = generate_embedding(query)

results = retrieve_documents(
    index=loaded_index,
    query_embedding=query_embedding,
    documents=loaded_documents,
    top_k=3
)


# STEP 7: Display results

print("\nSEARCH RESULTS AFTER LOADING")

for rank, result in enumerate(results, start=1):

    print("\n" + "=" * 50)

    print("RANK:", rank)
    print("CHUNK ID:", result["chunk_id"])
    print("SOURCE:", result["source"])
    print("PAGE:", result["page_number"])

    print(
        "SCORE:",
        round(result["similarity_score"], 4)
    )

    print("TEXT:", result["chunk_text"])
