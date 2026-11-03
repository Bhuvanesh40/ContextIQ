
import json
from pathlib import Path

import faiss


def save_vectorstore(
    index,
    documents,
    index_path,
    metadata_path
):
    # Persist the FAISS index and its parallel document metadata JSON.
    # One metadata record per vector is required to map search hits to text.
    if index.ntotal != len(documents):
        raise ValueError(
            "FAISS vector count and metadata count do not match"
        )

    # Path objects provide consistent parent-directory and file operations.
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    # Create missing output folders so first-time saves succeed.
    index_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # FAISS uses its own binary format for the vector index.
    faiss.write_index(
        index,
        str(index_path)
    )

    # JSON keeps source names, page numbers, IDs, and chunk text human-readable.
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Vector store saved successfully!")


def load_vectorstore(index_path, metadata_path):
    # Load a persisted index/metadata pair and verify their positional alignment.
    # Normalize both paths before performing file checks.
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    # Report which persisted component is missing rather than returning partial state.
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    # Reconstruct the binary vector index.
    index = faiss.read_index(
        str(index_path)
    )

    # Load document records in the same order used when the index was built.
    with metadata_path.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    # A count mismatch means vector positions can no longer identify metadata safely.
    if index.ntotal != len(documents):
        raise ValueError(
            "Loaded index and metadata counts do not match"
        )

    print("Vector store loaded successfully!")

    return index, documents
