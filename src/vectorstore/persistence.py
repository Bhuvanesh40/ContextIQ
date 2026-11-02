
import json
from pathlib import Path

import faiss


def save_vectorstore(
    index,
    documents,
    index_path,
    metadata_path
):

    # Step 1: Verify that vector and metadata counts match
    if index.ntotal != len(documents):
        raise ValueError(
            "FAISS vector count and metadata count do not match"
        )

    # Step 2: Convert paths into Path objects
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    # Step 3: Create parent folders if necessary
    index_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Step 4: Save the FAISS index
    faiss.write_index(
        index,
        str(index_path)
    )

    # Step 5: Save document metadata as JSON
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Vector store saved successfully!")


def load_vectorstore(index_path, metadata_path):

    # Step 1: Convert paths into Path objects
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    # Step 2: Check whether files exist
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    # Step 3: Load FAISS index
    index = faiss.read_index(
        str(index_path)
    )

    # Step 4: Load metadata
    with metadata_path.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    # Step 5: Verify alignment
    if index.ntotal != len(documents):
        raise ValueError(
            "Loaded index and metadata counts do not match"
        )

    print("Vector store loaded successfully!")

    return index, documents
