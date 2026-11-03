from pathlib import Path
import sys
import json

# Allow imports from src
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import load_knowledge_base, search_context
from evaluation.evaluation_dataset import evaluation_dataset
from evaluation.retrieval_evaluator import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k,
    ndcg_at_k
)


def evaluate_retrieval(index, documents, dataset, k=3, min_similarity=0.0):
    # Score retrieval for each labeled query and return means plus details.
    all_results = []

    # Each dataset item supplies one question and its ground-truth chunk IDs.
    for item in dataset:
        question = item["question"]
        relevant_ids = set(item["relevant_chunk_ids"])

        print("\n" + "=" * 60)
        print("QUESTION:", question)

        # Reuse the production retrieval path so evaluation reflects real behavior.
        results = search_context(
            index=index,
            documents=documents,
            question=question,
            top_k=k,
            min_similarity=min_similarity
        )

        # Metrics compare stable chunk IDs rather than chunk text.
        retrieved_ids = [
            document["chunk_id"]
            for document in results
        ]

        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        recall = recall_at_k(retrieved_ids, relevant_ids, k)
        mrr = reciprocal_rank(retrieved_ids, relevant_ids, k)
        hit_rate = hit_rate_at_k(retrieved_ids, relevant_ids, k)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k)

        all_results.append({
            "question": question,
            "retrieved_ids": retrieved_ids,
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "hit_rate": hit_rate,
            "ndcg": ndcg
        })

        print("\nMETRICS")
        print(f"Precision@{k}: {precision:.4f}")
        print(f"Recall@{k}:    {recall:.4f}")
        print(f"MRR@{k}:       {mrr:.4f}")
        print(f"Hit Rate@{k}:  {hit_rate:.4f}")
        print(f"NDCG@{k}:      {ndcg:.4f}")

    # Avoid dividing by zero if callers pass an empty evaluation dataset.
    count = len(all_results)
    if count == 0:
        raise ValueError("Evaluation dataset is empty")

    summary = {
        "questions_evaluated": count,
        "k": k,
        "min_similarity": min_similarity,
        "average_precision": sum(r["precision"] for r in all_results) / count,
        "average_recall": sum(r["recall"] for r in all_results) / count,
        "average_mrr": sum(r["mrr"] for r in all_results) / count,
        "average_hit_rate": sum(r["hit_rate"] for r in all_results) / count,
        "average_ndcg": sum(r["ndcg"] for r in all_results) / count
    }

    return summary, all_results


def main():
    # Run baseline and thresholded retrieval, then save an evaluation report.
    # Load the same persisted knowledge base used by the app.
    print("\nLOADING KNOWLEDGE BASE")
    index, documents = load_knowledge_base()

    # Baseline keeps the top-k ranking regardless of score.
    print("\n\nEVALUATION 1: WITHOUT SIMILARITY FILTER")
    baseline, baseline_details = evaluate_retrieval(
        index, documents, evaluation_dataset,
        k=3,
        min_similarity=0.0
    )

    # Filtered results use the app's current minimum similarity cutoff.
    print("\n\nEVALUATION 2: WITH SIMILARITY FILTER (threshold=0.20)")
    filtered, filtered_details = evaluate_retrieval(
        index, documents, evaluation_dataset,
        k=3,
        min_similarity=0.20
    )

    print("\n" + "=" * 60)
    print("CONTEXTIQ RETRIEVAL EVALUATION REPORT")
    print("=" * 60)

    print("\nBASELINE (k=3, min_similarity=0.0):")
    for metric, value in baseline.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

    print("\nTHRESHOLD FILTERED (k=3, min_similarity=0.20):")
    for metric, value in filtered.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

    # Store aggregate and query-level results for later dashboard inspection.
    report = {
        "baseline": baseline,
        "filtered": filtered,
        "baseline_details": baseline_details,
        "filtered_details": filtered_details
    }

    output_path = SRC_DIR.parent / "data" / "index" / "evaluation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nEvaluation report saved to:", output_path)


if __name__ == "__main__":
    main()
