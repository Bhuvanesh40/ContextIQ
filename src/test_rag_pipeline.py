from rag_pipeline import detect_knowledge_gap
from pathlib import Path

from rag_pipeline import (
    process_pdf,
    load_knowledge_base,
    search_context
)


# STEP 1: Check whether the knowledge base already exists

index_path = Path("data/index/faiss.index")
metadata_path = Path("data/index/metadata.json")

if index_path.exists() and metadata_path.exists():

    print("\nEXISTING KNOWLEDGE BASE FOUND")

    # Load previously saved index
    index, documents = load_knowledge_base()

else:

    print("\nNO SAVED KNOWLEDGE BASE FOUND")

    # Process PDF and create knowledge base
    pdf_path = "data/uploads/sample.pdf"

    index, documents = process_pdf(pdf_path)


# STEP 2: Ask a question

question = "What is Retrieval-Augmented Generation?"


# STEP 3: Search the knowledge base

results = search_context(
    index=index,
    documents=documents,
    question=question,
    top_k=3
)
