import math


def precision_at_k(retrieved_ids, relevant_ids, k):
    # Measure the fraction of the first k results that are relevant.
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("Relevant chunk IDs cannot be empty")

    # Restrict scoring to the requested result prefix.
    top_k_results = retrieved_ids[:k]

    # An empty result list has no relevant hits by definition.
    if not top_k_results:
        return 0.0

    relevant_count = sum(
        1 for chunk_id in top_k_results
        if chunk_id in relevant_ids
    )

    return relevant_count / len(top_k_results)


def recall_at_k(retrieved_ids, relevant_ids, k):
    # Measure the fraction of all relevant IDs found in the first k results.
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("Relevant chunk IDs cannot be empty")

    # Only retrieved IDs in the requested ranking window count as found.
    top_k_results = retrieved_ids[:k]

    relevant_count = sum(
        1 for chunk_id in top_k_results
        if chunk_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids, k=None):
    # Score the rank of the first relevant result (higher is better).
    if k is not None and k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("Relevant chunk IDs cannot be empty")

    # An optional k bounds the search; otherwise inspect the full result list.
    results = retrieved_ids if k is None else retrieved_ids[:k]

    # The first relevant result contributes its reciprocal one-based rank.
    for rank, chunk_id in enumerate(results, start=1):
        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


def hit_rate_at_k(retrieved_ids, relevant_ids, k):
    # Return 1 if any of the first k results is relevant, otherwise 0.
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("Relevant chunk IDs cannot be empty")

    # Only the first k retrieved IDs are considered for this binary metric.
    top_k_results = retrieved_ids[:k]

    for chunk_id in top_k_results:
        if chunk_id in relevant_ids:
            return 1

    return 0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """
    Calculate Normalized Discounted Cumulative Gain at rank k (NDCG@K)
    for binary relevance.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("Relevant chunk IDs cannot be empty")

    # Discount relevant hits by rank so earlier matches score more highly.
    top_k_results = retrieved_ids[:k]
    if not top_k_results:
        return 0.0

    dcg = 0.0
    for rank, chunk_id in enumerate(top_k_results, start=1):
        if chunk_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Compute the best possible discounted score for the available labels.
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    # Normalize against the ideal ordering, yielding a score between 0 and 1.
    return dcg / idcg
