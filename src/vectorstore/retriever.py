
from vectorstore.faiss_store import search_faiss


def retrieve_documents(
    index,
    query_embedding,
    documents,
    top_k=3,
    min_similarity=0.20,
    source_filter=None,
    page_filter=None
):
    # Find relevant chunks and return their text plus citation metadata.
    # Reject page numbers that cannot identify a real one-based PDF page.
    if page_filter is not None and page_filter <= 0:
        raise ValueError("Page number must be greater than zero")

    # Search every indexed vector when filtering by metadata; otherwise a
    # source/page filter could discard the global top-k and miss valid matches.
    search_k = index.ntotal if (
        source_filter is not None or page_filter is not None
    ) else top_k

    scores, indices = search_faiss(
        index,
        query_embedding,
        search_k
    )

    # Convert FAISS's vector positions into document records, applying threshold
    # and metadata filters before collecting the final top_k results.
    retrieved_documents = []

    for score, index_position in zip(scores, indices):

        # FAISS uses -1 when fewer than k neighbors are available.
        if index_position < 0:
            continue

        score = float(score)

        # Lower-scoring matches are considered too weak to use as evidence.
        if score < min_similarity:
            continue

        # Vector order and metadata order are kept aligned by ingestion/persistence.
        document = documents[int(index_position)]

        # Filter by source PDF
        if (
            source_filter is not None
            and document["source"].lower() != source_filter.lower()
        ):
            continue

        # Filter by page number
        if (
            page_filter is not None
            and document["page_number"] != page_filter
        ):
            continue

        # Return only the fields needed by prompts, citations, and the UI.
        retrieved_documents.append({
            "chunk_id": document["chunk_id"],
            "source": document["source"],
            "page_number": document["page_number"],
            "chunk_text": document["chunk_text"],
            "similarity_score": score
        })

        if len(retrieved_documents) >= top_k:
            break

    return retrieved_documents
