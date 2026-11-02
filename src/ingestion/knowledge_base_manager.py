from pathlib import Path
import numpy as np
import faiss

from ingestion.document_loader import extract_pdf
from preprocessing.text_cleaner import clean_text
from chunking.recursive_chunking import recursive_chunk_text
from metadata.metadata_manager import attach_metadata
from embeddings.embedding_model import generate_embeddings
from vectorstore.persistence import (
    save_vectorstore,
    load_vectorstore
)
from vectorstore.faiss_store import create_faiss_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = str(PROJECT_ROOT / "data" / "index" / "faiss.index")
METADATA_PATH = str(PROJECT_ROOT / "data" / "index" / "metadata.json")


def clear_knowledge_base():
    """Clear all indexed vectors and metadata."""
    index_file = Path(INDEX_PATH)
    metadata_file = Path(METADATA_PATH)

    if index_file.exists():
        index_file.unlink()
    if metadata_file.exists():
        metadata_file.unlink()

    return {"status": "cleared", "message": "Knowledge base reset successfully."}


def ingest_pdf(pdf_path, chunk_size=1000, overlap=200):
    """
    Add a PDF to the existing ContextIQ knowledge base.
    Returns the updated FAISS index and metadata.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Load existing knowledge base if it exists.
    index_file = Path(INDEX_PATH)
    metadata_file = Path(METADATA_PATH)

    if index_file.exists() and metadata_file.exists():
        index, documents = load_vectorstore(
            INDEX_PATH,
            METADATA_PATH
        )
    else:
        index = None
        documents = []

    # Prevent indexing the same filename twice.
    existing_sources = {
        document["source"].lower()
        for document in documents
    }

    if pdf_path.name.lower() in existing_sources:
        return {
            "status": "already_exists",
            "message": f"{pdf_path.name} is already indexed.",
            "index": index,
            "documents": documents
        }

    pages = extract_pdf(pdf_path)

    new_documents = []
    next_chunk_index = len(documents)

    for page in pages:
        cleaned_text = clean_text(page["text"])

        if not cleaned_text:
            continue

        chunks = recursive_chunk_text(
            cleaned_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        page_documents = attach_metadata(
            chunks=chunks,
            source=page["source"],
            page_number=page["page_number"],
            start_index=next_chunk_index
        )

        new_documents.extend(page_documents)
        next_chunk_index += len(page_documents)

    if not new_documents:
        raise ValueError("No usable text chunks were extracted.")

    texts = [
        document["chunk_text"]
        for document in new_documents
    ]

    embeddings = generate_embeddings(texts)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Create a new index or append to the existing one.
    if index is None:
        index = create_faiss_index(embeddings)
    else:
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

    documents.extend(new_documents)

    save_vectorstore(
        index,
        documents,
        INDEX_PATH,
        METADATA_PATH
    )

    return {
        "status": "success",
        "message": (
            f"Successfully indexed {pdf_path.name}. "
            f"Added {len(new_documents)} chunks."
        ),
        "index": index,
        "documents": documents
    }


def ingest_multiple_pdfs(pdf_paths, chunk_size=1000, overlap=200):
    """
    Ingest a batch of multiple PDF documents sequentially into the knowledge base.
    """
    results = []
    last_index = None
    last_documents = []

    for path in pdf_paths:
        try:
            res = ingest_pdf(path, chunk_size=chunk_size, overlap=overlap)
            results.append(res)
            last_index = res.get("index")
            last_documents = res.get("documents", [])
        except Exception as e:
            results.append({
                "status": "error",
                "message": f"Failed to index {Path(path).name}: {str(e)}"
            })

    return {
        "results": results,
        "index": last_index,
        "documents": last_documents
    }
