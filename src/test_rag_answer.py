from rag_pipeline import (
    load_knowledge_base,
    generate_rag_answer
)

# Load existing FAISS knowledge base
print("\nLOADING KNOWLEDGE BASE...")

index, documents = load_knowledge_base()

# Ask a question about your PDFs
question = "What is Retrieval-Augmented Generation?"

print("\nQUESTION:")
print(question)

# Generate RAG answer
result = generate_rag_answer(
    index=index,
    documents=documents,
    question=question,
    top_k=3
)

# Display answer
print("\nCONTEXTIQ ANSWER:")
print(result["answer"])

# Display sources
print("\nSOURCES USED:")

for source in result["sources"]:
    print(
        f"- {source['source']} | "
        f"Page: {source['page_number']} | "
        f"Similarity: {source['similarity_score']:.4f}"
    )