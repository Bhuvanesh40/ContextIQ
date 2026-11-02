
from pathlib import Path
import sys
import json

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from ingestion.document_loader import extract_pdf
from preprocessing.text_cleaner import clean_text
from chunking.recursive_chunking import recursive_chunk_text
from metadata.metadata_manager import attach_metadata
from embeddings.embedding_model import generate_embeddings
from vectorstore.faiss_store import create_faiss_index
from vectorstore.retriever import retrieve_documents

from evaluation.evaluation_dataset import evaluation_dataset
from evaluation.retrieval_evaluator import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k
)


PDF_DIRECTORY = Path("data/uploads")

CONFIGURATIONS = [
    (300, 50),
    (500, 50),
    (500, 100),
    (800, 100)
]


def build_temporary_index(chunk_size, overlap):

    all_documents = []

    pdf_files = sorted(PDF_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDF files found in data/uploads")

    for pdf_path in pdf_files:

        pages = extract_pdf(pdf_path)

        for page in pages:

            cleaned_text = clean_text(page["text"])

            chunks = recursive_chunk_text(
                cleaned_text,
                chunk_size=chunk_size,
                overlap=overlap
            )

            # Use unique IDs for each temporary index
            start_index = len(all_documents)

            page_documents = attach_metadata(
                chunks=chunks,
                source=page["source"],
                page_number=page["page_number"],
                start_index=start_index
            )

            all_documents.extend(page_documents)

    if not all_documents:
        raise ValueError("No chunks were generated")

    texts = [
        document["chunk_text"]
        for document in all_documents
    ]

    embeddings = generate_embeddings(texts)

    index = create_faiss_index(embeddings)

    return index, all_documents


def evaluate_configuration(index, documents):

    scores = {
        "precision": [],
        "recall": [],
        "mrr": [],
        "hit_rate": []
    }

    for item in evaluation_dataset:

        from embeddings.embedding_model import generate_embedding

        query_embedding = generate_embedding(item["question"])

        results = retrieve_documents(
            index=index,
            query_embedding=query_embedding,
            documents=documents,
            top_k=3,
            min_similarity=0.0
        )

        retrieved_ids = [
            document["chunk_id"]
            for document in results
        ]

        relevant_ids = set(item["relevant_chunk_ids"])

        scores["precision"].append(
            precision_at_k(retrieved_ids, relevant_ids, 3)
        )

        scores["recall"].append(
            recall_at_k(retrieved_ids, relevant_ids, 3)
        )

        scores["mrr"].append(
            reciprocal_rank(retrieved_ids, relevant_ids, 3)
        )

        scores["hit_rate"].append(
            hit_rate_at_k(retrieved_ids, relevant_ids, 3)
        )

    count = len(evaluation_dataset)

    return {
        metric: sum(values) / count
        for metric, values in scores.items()
    }


def main():

    if not evaluation_dataset:
        raise ValueError("Evaluation dataset is empty")

    experiment_results = []

    for chunk_size, overlap in CONFIGURATIONS:

        print("\n" + "=" * 55)
        print("CHUNK SIZE:", chunk_size)
        print("OVERLAP:", overlap)

        index, documents = build_temporary_index(
            chunk_size,
            overlap
        )

        scores = evaluate_configuration(index, documents)

        result = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "total_chunks": len(documents),
            **scores
        }

        experiment_results.append(result)

        print("Total chunks:", len(documents))
        print("Precision:", round(scores["precision"], 4))
        print("Recall:", round(scores["recall"], 4))
        print("MRR:", round(scores["mrr"], 4))
        print("Hit Rate:", round(scores["hit_rate"], 4))

    output_path = Path(
        "data/index/chunking_experiment_report.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(experiment_results, file, indent=4)

    print("\nCHUNKING EXPERIMENT COMPLETED")
    print("Report saved:", output_path)


if __name__ == "__main__":
    main()
    