import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root and src/ are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def get_evaluator_models():
    """Initialize Gemini LLM and SentenceTransformer embedding evaluator models."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for RAGAS evaluation.")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        api_key=api_key,
        temperature=0.0
    )
    ragas_llm = LangchainLLMWrapper(llm)

    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    ragas_emb = LangchainEmbeddingsWrapper(emb)

    return ragas_llm, ragas_emb


def build_evaluation_records(index, documents, dataset, top_k=3, min_similarity=0.20):
    """
    Execute ContextIQ RAG pipeline for all evaluation queries and format
    into the RAGAS dataset schema (supporting both standard and v0.4+ column names).
    """
    from rag_pipeline import generate_rag_answer

    questions = []
    contexts = []
    answers = []
    ground_truths = []

    print("\n" + "=" * 60)
    print("RUNNING CONTEXTIQ INFERENCE ON EVALUATION QUERIES")
    print("=" * 60)

    for idx, item in enumerate(dataset, start=1):
        q = item["question"]
        gt = item.get("ground_truth", "")

        print(f"\n[{idx}/{len(dataset)}] Query: {q}")
        res = generate_rag_answer(
            index=index,
            documents=documents,
            question=q,
            top_k=top_k,
            min_similarity=min_similarity
        )

        retrieved_texts = [
            src["chunk_text"] for src in res.get("sources", [])
        ]
        if not retrieved_texts:
            retrieved_texts = ["No relevant context found in documents."]

        questions.append(q)
        contexts.append(retrieved_texts)
        answers.append(res.get("answer", ""))
        ground_truths.append(gt)

    return {
        "user_input": questions,
        "question": questions,
        "retrieved_contexts": contexts,
        "contexts": contexts,
        "response": answers,
        "answer": answers,
        "reference": ground_truths,
        "ground_truth": ground_truths
    }


def evaluate_ragas(index, documents, dataset, top_k=3, min_similarity=0.20):
    """
    Run complete RAGAS generation and retrieval evaluation.
    Returns average metrics dictionary and detailed records.
    """
    ragas_llm, ragas_emb = get_evaluator_models()

    eval_data = build_evaluation_records(
        index=index,
        documents=documents,
        dataset=dataset,
        top_k=top_k,
        min_similarity=min_similarity
    )

    hf_dataset = Dataset.from_dict(eval_data)

    print("\n" + "=" * 60)
    print("CALCULATING RAGAS METRICS WITH GEMINI JUDGE")
    print("=" * 60)
    print("Evaluating: Faithfulness, Answer Relevance, Context Precision, Context Recall, Answer Correctness...")

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    ]

    results = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb
    )

    return results, eval_data


def main():
    from rag_pipeline import load_knowledge_base
    from evaluation.evaluation_dataset import evaluation_dataset

    print("\nLOADING KNOWLEDGE BASE...")
    index, documents = load_knowledge_base()

    results, eval_data = evaluate_ragas(
        index=index,
        documents=documents,
        dataset=evaluation_dataset,
        top_k=3,
        min_similarity=0.20
    )

    print("\n" + "=" * 60)
    print("CONTEXTIQ RAGAS GENERATION EVALUATION REPORT")
    print("=" * 60)

    summary = {}
    # Extract results as dict or dataframe
    if hasattr(results, "to_pandas"):
        df = results.to_pandas()
        numeric_cols = [c for c in df.columns if c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]]
        for col in numeric_cols:
            score = float(df[col].mean())
            summary[col] = round(score, 4)
            print(f"  {col}: {summary[col]:.4f}")
    else:
        for metric_name, score in results.items():
            summary[metric_name] = round(float(score), 4) if score is not None else 0.0
            print(f"  {metric_name}: {summary[metric_name]:.4f}")

    # Export report to disk
    output_path = PROJECT_ROOT / "data" / "index" / "ragas_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "summary": summary,
        "queries_evaluated": len(eval_data["question"]),
        "records": [
            {
                "question": eval_data["question"][i],
                "ground_truth": eval_data["ground_truth"][i],
                "generated_answer": eval_data["answer"][i],
                "contexts": eval_data["contexts"][i]
            }
            for i in range(len(eval_data["question"]))
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=4)

    print(f"\nRAGAS Evaluation report saved to: {output_path}")


if __name__ == "__main__":
    main()
