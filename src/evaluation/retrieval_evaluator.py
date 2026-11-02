import math


def precision_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("Relevant chunk IDs cannot be empty")

    top_k_results = retrieved_ids[:k]

    if not top_k_results:
        return 0.0

    relevant_count = sum(
        1 for chunk_id in top_k_results
        if chunk_id in relevant_ids
    )

    return relevant_count / len(top_k_results)


def recall_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("Relevant chunk IDs cannot be empty")

    top_k_results = retrieved_ids[:k]

    relevant_count = sum(
        1 for chunk_id in top_k_results
        if chunk_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids, k=None):
    if k is not None and k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("Relevant chunk IDs cannot be empty")

    results = retrieved_ids if k is None else retrieved_ids[:k]

    for rank, chunk_id in enumerate(results, start=1):
        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


def hit_rate_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("Relevant chunk IDs cannot be empty")

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

    top_k_results = retrieved_ids[:k]
    if not top_k_results:
        return 0.0

    dcg = 0.0
    for rank, chunk_id in enumerate(top_k_results, start=1):
        if chunk_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
