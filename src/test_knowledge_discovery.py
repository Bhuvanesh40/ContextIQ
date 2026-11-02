from rag_pipeline import (
    load_knowledge_base,
    generate_cross_document_answer,
    generate_document_comparison,
    generate_related_concepts,
    detect_knowledge_gap
)


def print_result(title, result, answer_key):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(result[answer_key])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(
            f"- {source['source']} | "
            f"Page {source['page_number']} | "
            f"Score: {source['similarity_score']:.4f}"
        )


def main():
    
    index, documents = load_knowledge_base()

    available_sources = sorted({
        document["source"] for document in documents
    })

    print("AVAILABLE DOCUMENTS:")
    for source in available_sources:
        print("-", source)

    if len(available_sources) < 2:
        print("\nUpload at least two PDFs before testing document comparison.")
        return
    result = detect_knowledge_gap(
    index=index,
    documents=documents,
    question="Explain quantum computing hardware in detail.",
    top_k=8,
    min_similarity=0.30
)

    print("\n" + "=" * 70)
    print("TEST 4: KNOWLEDGE GAP DETECTION")
    print("=" * 70)

    print("Status:", result["status"])
    print("Explanation:", result["explanation"])
    print("Retrieved chunks:", result["retrieved_chunk_count"])
    print("Source documents:", result["source_documents"])

    # TEST 1: CROSS-DOCUMENT QUESTION ANSWERING
    result = generate_cross_document_answer(
        index=index,
        documents=documents,
        question="How are embeddings and large language models useful in RAG?",
        top_k=8
    )

    print_result(
        "TEST 1: CROSS-DOCUMENT ANSWERING",
        result,
        "answer"
    )

    # TEST 2: DOCUMENT COMPARISON
    source_a = "sample.pdf"
    source_b = "llm_guide.pdf"
    result = generate_document_comparison(
        index=index,
        documents=documents,
        question="What role do large language models play in RAG?",
        source_a=source_a,
        source_b=source_b,
        top_k=5
    )

    print_result(
        "TEST 2: DOCUMENT COMPARISON",
        result,
        "comparison"
    )

    # TEST 3: RELATED CONCEPT DISCOVERY
    result = generate_related_concepts(
        index=index,
        documents=documents,
        question="What should I understand before learning Retrieval-Augmented Generation?",
        top_k=8
    )

    print_result(
        "TEST 3: RELATED CONCEPT DISCOVERY",
        result,
        "concepts"
    )


if __name__ == "__main__":
    main()