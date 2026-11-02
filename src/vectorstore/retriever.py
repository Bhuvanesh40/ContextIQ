
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
    # Validate metadata filters
    if page_filter is not None and page_filter <= 0:
        raise ValueError("Page number must be greater than zero")

    # If filters are provided, retrieve all candidates so
    # filtering does not accidentally remove the best matches.
    search_k = index.ntotal if (
        source_filter is not None or page_filter is not None
    ) else top_k

    scores, indices = search_faiss(
        index,
        query_embedding,
        search_k
    )

    retrieved_documents = []

    for score, index_position in zip(scores, indices):

        if index_position < 0:
            continue

        score = float(score)

        if score < min_similarity:
            continue

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
