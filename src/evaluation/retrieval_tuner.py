
from pathlib import Path
import sys
import json

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import load_knowledge_base, search_context
from evaluation.evaluation_dataset import evaluation_dataset

from evaluation.retrieval_evaluator import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k
)


def evaluate_configuration(
    index,
    documents,
    dataset,
    top_k,
    threshold
):
    metrics = {
        "precision": [],
        "recall": [],
        "mrr": [],
        "hit_rate": []
    }

    for item in dataset:

        results = search_context(
            index=index,
            documents=documents,
            question=item["question"],
            top_k=top_k,
            min_similarity=threshold
        )

        retrieved_ids = [
            document["chunk_id"]
            for document in results
        ]

        relevant_ids = set(item["relevant_chunk_ids"])

        metrics["precision"].append(
            precision_at_k(retrieved_ids, relevant_ids, top_k)
        )

        metrics["recall"].append(
            recall_at_k(retrieved_ids, relevant_ids, top_k)
        )

        metrics["mrr"].append(
            reciprocal_rank(retrieved_ids, relevant_ids, top_k)
        )

        metrics["hit_rate"].append(
            hit_rate_at_k(retrieved_ids, relevant_ids, top_k)
        )

    count = len(dataset)

    return {
        metric: sum(values) / count
        for metric, values in metrics.items()
    }


def main():

    index, documents = load_knowledge_base()

    if not evaluation_dataset:
        raise ValueError("Evaluation dataset is empty")

    # Experiment configurations
    top_k_values = [1, 3, 5]
    thresholds = [0.0, 0.10, 0.20, 0.30, 0.40]

    experiment_results = []

    total_experiments = len(top_k_values) * len(thresholds)
    experiment_number = 0

    for top_k in top_k_values:

        for threshold in thresholds:

            experiment_number += 1

            print(
                f"\nExperiment {experiment_number}/{total_experiments}"
            )
            print(f"Top-K: {top_k}")
            print(f"Threshold: {threshold}")

            scores = evaluate_configuration(
                index=index,
                documents=documents,
                dataset=evaluation_dataset,
                top_k=top_k,
                threshold=threshold
            )

            result = {
                "top_k": top_k,
                "threshold": threshold,
                **scores
            }

            experiment_results.append(result)

            print("Precision:", round(scores["precision"], 4))
            print("Recall:", round(scores["recall"], 4))
            print("MRR:", round(scores["mrr"], 4))
            print("Hit Rate:", round(scores["hit_rate"], 4))

    # Save all experiments
    output_path = Path("data/index/retrieval_tuning_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(experiment_results, file, indent=4)

    # Rank configurations by a simple balanced score
    best = max(
        experiment_results,
        key=lambda item: (
            item["precision"] +
            item["recall"] +
            item["mrr"] +
            item["hit_rate"]
        ) / 4
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL TUNING SUMMARY")
    print("=" * 60)

    print("Best configuration in this experiment:")
    print("Top-K:", best["top_k"])
    print("Threshold:", best["threshold"])

    for metric in ["precision", "recall", "mrr", "hit_rate"]:
        print(f"{metric}: {best[metric]:.4f}")

    print("\nReport saved:", output_path)


if __name__ == "__main__":
    main()
