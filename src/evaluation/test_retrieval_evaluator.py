from retrieval_evaluator import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k,
    ndcg_at_k
)

# Simulated FAISS retrieval results
retrieved_ids = [
    "chunk_0001",
    "chunk_0005",
    "chunk_0004",
    "chunk_0002",
    "chunk_0003"
]

# Ground truth relevant chunks
relevant_ids = {
    "chunk_0001",
    "chunk_0002",
    "chunk_0003",
    "chunk_0006"
}

precision = precision_at_k(retrieved_ids=retrieved_ids, relevant_ids=relevant_ids, k=5)
recall = recall_at_k(retrieved_ids=retrieved_ids, relevant_ids=relevant_ids, k=5)
mrr = reciprocal_rank(retrieved_ids=retrieved_ids, relevant_ids=relevant_ids, k=5)
hit_rate = hit_rate_at_k(retrieved_ids=retrieved_ids, relevant_ids=relevant_ids, k=5)
ndcg = ndcg_at_k(retrieved_ids=retrieved_ids, relevant_ids=relevant_ids, k=5)

print("=" * 40)
print("RETRIEVAL EVALUATOR METRIC TESTS (K=5)")
print("=" * 40)
print(f"Precision@5:      {precision:.4f} ({precision * 100:.1f}%)")
print(f"Recall@5:         {recall:.4f} ({recall * 100:.1f}%)")
print(f"Reciprocal Rank:  {mrr:.4f}")
print(f"Hit Rate@5:       {hit_rate}")
print(f"NDCG@5:           {ndcg:.4f}")
print("=" * 40)
