from pathlib import Path

from llm.gemini_client import generate_response
from llm.prompt_builder import build_rag_prompt
from ingestion.document_loader import extract_pdf
from preprocessing.text_cleaner import clean_text
from chunking.recursive_chunking import recursive_chunk_text
from llm.conversation_manager import rewrite_question
from metadata.metadata_manager import attach_metadata
from llm.knowledge_discovery import (
    answer_across_documents,
    compare_documents,
    discover_related_concepts
)

from embeddings.embedding_model import (
    generate_embedding,
    generate_embeddings
)

from vectorstore.faiss_store import create_faiss_index
from vectorstore.retriever import retrieve_documents

from vectorstore.persistence import (
    save_vectorstore,
    load_vectorstore
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = str(PROJECT_ROOT / "data" / "index" / "faiss.index")
METADATA_PATH = str(PROJECT_ROOT / "data" / "index" / "metadata.json")


def process_pdf(pdf_path, chunk_size=1000, overlap=200):

    print("\nSTEP 1: EXTRACTING PDF")

    pages = extract_pdf(pdf_path)

    print("Pages extracted:", len(pages))

    all_documents = []
    next_chunk_index = 0

    for page in pages:

        print("\nProcessing page:", page["page_number"])

        # STEP 2: Clean text
        cleaned_text = clean_text(page["text"])

        # STEP 3: Create chunks
        chunks = recursive_chunk_text(
            cleaned_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        # STEP 4: Attach metadata
        documents = attach_metadata(
            chunks=chunks,
            source=page["source"],
            page_number=page["page_number"],
            start_index=next_chunk_index
        )

        all_documents.extend(documents)

        next_chunk_index += len(documents)

    if not all_documents:
        raise ValueError("No usable text chunks were generated")

    print("\nTOTAL CHUNKS:", len(all_documents))

    # STEP 5: Generate embeddings
    print("\nSTEP 5: GENERATING EMBEDDINGS")

    chunk_texts = [
        document["chunk_text"]
        for document in all_documents
    ]

    embeddings = generate_embeddings(chunk_texts)

    print("Embedding shape:", embeddings.shape)

    # STEP 6: Create FAISS index
    print("\nSTEP 6: CREATING FAISS INDEX")

    index = create_faiss_index(embeddings)

    print("Total indexed vectors:", index.ntotal)

    # STEP 7: Save index and metadata
    print("\nSTEP 7: SAVING VECTOR STORE")

    save_vectorstore(
        index=index,
        documents=all_documents,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )

    return index, all_documents


def load_knowledge_base():

    print("\nLOADING SAVED KNOWLEDGE BASE")

    index, documents = load_vectorstore(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )

    print("Total vectors:", index.ntotal)
    print("Total metadata records:", len(documents))

    return index, documents



def search_context(
    index,
    documents,
    question,
    top_k=3,
    min_similarity=0.20,
    source_filter=None,
    page_filter=None
):
    query_embedding = generate_embedding(question)

    results = retrieve_documents(
        index=index,
        query_embedding=query_embedding,
        documents=documents,
        top_k=top_k,
        min_similarity=min_similarity,
        source_filter=source_filter,
        page_filter=page_filter
    )

    print("\nSEARCH QUESTION:", question)
    print("\nRETRIEVED RESULTS")

    if not results:
        print("No matching documents found.")

    for rank, document in enumerate(results, start=1):
        print("\n" + "=" * 50)
        print("RANK:", rank)
        print("CHUNK ID:", document["chunk_id"])
        print("SOURCE:", document["source"])
        print("PAGE:", document["page_number"])
        print("SIMILARITY:", round(document["similarity_score"], 4))
        print("CONTENT:")
        print(document["chunk_text"])

    return results


def generate_rag_answer(
    index,
    documents,
    question,
    top_k=3,
    min_similarity=0.20,
    chat_history=None,
    source_filter=None,
    page_filter=None
):

    if chat_history is None:
        chat_history = []

    # STEP 1: Rewrite the follow-up question
    standalone_question = rewrite_question(
        question=question,
        chat_history=chat_history
    )

    print("\nORIGINAL QUESTION:", question)
    print("REWRITTEN QUESTION:", standalone_question)

    # STEP 2: Retrieve using the standalone question
    retrieved_documents = search_context(
        index=index,
        documents=documents,
        question=standalone_question,
        top_k=top_k,
        min_similarity=min_similarity,
        source_filter=source_filter,
        page_filter=page_filter
    )

    # STEP 3: Handle missing context
    if not retrieved_documents:

        answer = (
            "The available documents do not contain "
            "enough relevant information to answer "
            "this question."
        )

    else:

        # STEP 4: Build context-aware prompt
        prompt = build_rag_prompt(
            question=question,
            retrieved_documents=retrieved_documents,
            chat_history=chat_history
        )

        # STEP 5: Generate answer
        answer = generate_response(prompt)

    return {
        "question": question,
        "standalone_question": standalone_question,
        "answer": answer,
        "sources": retrieved_documents
    }
def generate_cross_document_answer(
    index,
    documents,
    question,
    top_k=8,
    min_similarity=0.20
):
    """Answer a question using relevant chunks across all PDFs."""

    retrieved_documents = search_context(
        index=index,
        documents=documents,
        question=question,
        top_k=top_k,
        min_similarity=min_similarity
    )

    if not retrieved_documents:
        return {
            "question": question,
            "answer": "I couldn't find enough relevant information in your uploaded documents.",
            "sources": []
        }

    answer = answer_across_documents(
        question,
        retrieved_documents
    )

    return {
        "question": question,
        "answer": answer,
        "sources": retrieved_documents
    }

def detect_knowledge_gap(
    index,
    documents,
    question,
    top_k=8,
    min_similarity=0.20,
    minimum_chunks=2
):
    """Check whether the knowledge base contains supporting evidence."""

    retrieved_documents = search_context(
        index=index,
        documents=documents,
        question=question,
        top_k=top_k,
        min_similarity=min_similarity
    )

    unique_sources = sorted({
        document["source"]
        for document in retrieved_documents
    })

    if not retrieved_documents:
        status = "No supporting evidence found"
        explanation = (
            "The uploaded documents did not return sufficiently "
            "similar passages for this question."
        )

    elif len(retrieved_documents) < minimum_chunks:
        status = "Limited supporting evidence"
        explanation = (
            "Some relevant information was found, but there may "
            "not be enough evidence for a complete answer."
        )

    else:
        status = "Supporting evidence found"
        explanation = (
            "Relevant passages were found. This does not guarantee "
            "that every aspect of the question is covered."
        )

    return {
        "question": question,
        "status": status,
        "explanation": explanation,
        "retrieved_chunk_count": len(retrieved_documents),
        "source_documents": unique_sources,
        "sources": retrieved_documents
    }
    
def generate_document_comparison(
    index,
    documents,
    question,
    source_a,
    source_b,
    top_k=5,
    min_similarity=0.20
):
    """Compare relevant information from two PDF documents."""

    available_sources = {
        document["source"].lower()
        for document in documents
    }

    if source_a.lower() not in available_sources:
        raise ValueError(f"Document not found: {source_a}")

    if source_b.lower() not in available_sources:
        raise ValueError(f"Document not found: {source_b}")

    if source_a.lower() == source_b.lower():
        raise ValueError("Choose two different documents.")

    results_a = search_context(
        index=index,
        documents=documents,
        question=question,
        top_k=top_k,
        min_similarity=min_similarity,
        source_filter=source_a
    )

    results_b = search_context(
        index=index,
        documents=documents,
        question=question,
        top_k=top_k,
        min_similarity=min_similarity,
        source_filter=source_b
    )

    comparison = compare_documents(
        question=question,
        documents_a=results_a,
        documents_b=results_b,
        source_a=source_a,
        source_b=source_b
    )

    return {
        "question": question,
        "document_a": source_a,
        "document_b": source_b,
        "comparison": comparison,
        "sources": results_a + results_b
    }


def generate_related_concepts(
    index,
    documents,
    question,
    top_k=8,
    min_similarity=0.20
):
    """Discover related concepts from uploaded documents."""

    retrieved_documents = search_context(
        index=index,
        documents=documents,
        question=question,
        top_k=top_k,
        min_similarity=min_similarity
    )

    if not retrieved_documents:
        return {
            "question": question,
            "concepts": "No relevant concepts found.",
            "sources": []
        }

    concepts = discover_related_concepts(
        question,
        retrieved_documents
    )

    return {
        "question": question,
        "concepts": concepts,
        "sources": retrieved_documents
    }

