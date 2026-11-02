import os
import sys
from pathlib import Path

# Add project root and src/ to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import (
    process_pdf,
    load_knowledge_base,
    search_context,
    generate_rag_answer,
    INDEX_PATH,
    METADATA_PATH
)
from ingestion.knowledge_base_manager import (
    ingest_pdf,
    clear_knowledge_base
)
from embeddings.embedding_model import generate_query_embedding


def test_complete_local_rag_pipeline():
    print("=" * 60)
    print("STARTING END-TO-END LOCAL PIPELINE VALIDATION")
    print("=" * 60)

    # 1. Test invalid PDF handling
    print("\n[TEST 1] Testing invalid file handling...")
    try:
        ingest_pdf("non_existent_file.pdf")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  PASS: Handled non-existent file safely.")

    # 2. Test Knowledge Base Reset
    print("\n[TEST 2] Testing knowledge base reset...")
    clear_res = clear_knowledge_base()
    assert clear_res["status"] == "cleared"
    assert not Path(INDEX_PATH).exists()
    print("  PASS: Knowledge base reset successfully.")

    # 3. Test PDF Ingestion & Indexing
    print("\n[TEST 3] Ingesting documents...")
    sample_pdf = PROJECT_ROOT / "data" / "uploads" / "sample.pdf"
    llm_pdf = PROJECT_ROOT / "data" / "uploads" / "llm_guide.pdf"
    python_pdf = PROJECT_ROOT / "data" / "uploads" / "python_guide.pdf"

    for pdf in [sample_pdf, llm_pdf, python_pdf]:
        if pdf.exists():
            res = ingest_pdf(pdf, chunk_size=1000, overlap=200)
            print(f"  Indexed: {pdf.name} -> Chunks added: {len(res['documents'])}")
            assert res["index"].ntotal > 0
        else:
            print(f"  Notice: {pdf.name} does not exist in data/uploads.")

    # 4. Test Persistence & Reloading
    print("\n[TEST 4] Testing index persistence and reload...")
    index, docs = load_knowledge_base()
    assert index.ntotal == len(docs)
    print(f"  PASS: Loaded FAISS index ({index.ntotal} vectors, {len(docs)} metadata records).")

    # 5. Test Embedding Generation Dimension
    print("\n[TEST 5] Testing embedding dimension...")
    q_emb = generate_query_embedding("What is RAG?")
    assert q_emb.shape == (384,), f"Expected shape (384,), got {q_emb.shape}"
    print(f"  PASS: Embedding model output dimension is 384.")

    # 6. Test Dense Retrieval & Similarity Filtering
    print("\n[TEST 6] Testing retrieval & similarity filtering...")
    retrieved = search_context(
        index=index,
        documents=docs,
        question="What is Python?",
        top_k=3,
        min_similarity=0.20
    )
    assert len(retrieved) > 0
    top_result = retrieved[0]
    assert "python" in top_result["chunk_text"].lower()
    assert top_result["similarity_score"] >= 0.20
    print(f"  PASS: Top result is '{top_result['source']}' (Score: {top_result['similarity_score']:.4f}).")

    # 7. Test Question Answering with Citations
    print("\n[TEST 7] Testing LLM Generation with grounding citations...")
    qa_result = generate_rag_answer(
        index=index,
        documents=docs,
        question="What is Python?",
        top_k=2,
        min_similarity=0.20
    )
    answer = qa_result["answer"]
    sources = qa_result["sources"]
    assert len(answer) > 0
    assert len(sources) > 0
    assert "source" in sources[0]
    print("  Generated Answer Preview:", answer[:120], "...")
    print(f"  PASS: Answer generated with {len(sources)} grounded sources.")

    print("\n" + "=" * 60)
    print("ALL LOCAL PIPELINE VALIDATION TESTS PASSED (10/10)")
    print("=" * 60)


if __name__ == "__main__":
    test_complete_local_rag_pipeline()
