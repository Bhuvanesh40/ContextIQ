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


# Keep index locations independent of the command's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = str(PROJECT_ROOT / "data" / "index" / "faiss.index")
METADATA_PATH = str(PROJECT_ROOT / "data" / "index" / "metadata.json")


def process_pdf(pdf_path, chunk_size=1000, overlap=200):
    # Build and persist a searchable vector store from one PDF.
    # Stage 1: extract page text while retaining source and page information.
    print("\nSTEP 1: EXTRACTING PDF")

    pages = extract_pdf(pdf_path)

    print("Pages extracted:", len(pages))

    all_documents = []
    next_chunk_index = 0

    # Process pages separately so every resulting chunk keeps its page number.
    for page in pages:

        print("\nProcessing page:", page["page_number"])

        # Stage 2: normalize extracted text to remove PDF artifacts/noise.
        cleaned_text = clean_text(page["text"])

        # Stage 3: split long page text into bounded overlapping passages.
        chunks = recursive_chunk_text(
            cleaned_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        # Stage 4: associate each passage with a stable ID and source location.
        documents = attach_metadata(
            chunks=chunks,
            source=page["source"],
            page_number=page["page_number"],
            start_index=next_chunk_index
        )

        all_documents.extend(documents)

        # Continue IDs after the last chunk generated on the preceding page.
        next_chunk_index += len(documents)

    # An empty index is not useful and would also break vector-search setup.
    if not all_documents:
        raise ValueError("No usable text chunks were generated")

    print("\nTOTAL CHUNKS:", len(all_documents))

    # Stage 5: map each chunk's meaning to a dense numerical vector.
    print("\nSTEP 5: GENERATING EMBEDDINGS")

    chunk_texts = [
        document["chunk_text"]
        for document in all_documents
    ]

    embeddings = generate_embeddings(chunk_texts)

    print("Embedding shape:", embeddings.shape)

    # Stage 6: normalize vectors and add them to the in-memory FAISS index.
    print("\nSTEP 6: CREATING FAISS INDEX")

    index = create_faiss_index(embeddings)

    print("Total indexed vectors:", index.ntotal)

    # Stage 7: persist vectors and their parallel metadata records together.
    print("\nSTEP 7: SAVING VECTOR STORE")

    save_vectorstore(
        index=index,
        documents=all_documents,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )

    return index, all_documents


def load_knowledge_base():
    # Load the persisted FAISS index and its aligned document metadata.
    print("\nLOADING SAVED KNOWLEDGE BASE")

    # Persistence validates that the vector count equals the metadata count.
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
    # Encode the question with the same model used for document chunks.
    query_embedding = generate_embedding(question)

    # Search FAISS and map matching vector positions back to document metadata.
    results = retrieve_documents(
        index=index,
        query_embedding=query_embedding,
        documents=documents,
        top_k=top_k,
        min_similarity=min_similarity,
        source_filter=source_filter,
        page_filter=page_filter
    )

    # Print the evidence for command-line runs and debugging; return it for the UI.
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
    # Retrieve supporting chunks and generate a conversational grounded answer.
    # Normalize absent history to an empty list for the rewriter and prompt builder.
    if chat_history is None:
        chat_history = []

    # Stage 1: resolve follow-up references before embedding/searching the query.
    standalone_question = rewrite_question(
        question=question,
        chat_history=chat_history
    )

    print("\nORIGINAL QUESTION:", question)
    print("REWRITTEN QUESTION:", standalone_question)

    # Stage 2: find the most similar chunks, respecting optional source/page filters.
    retrieved_documents = search_context(
        index=index,
        documents=documents,
        question=standalone_question,
        top_k=top_k,
        min_similarity=min_similarity,
        source_filter=source_filter,
        page_filter=page_filter
    )

    # Stage 3: do not call the LLM when retrieval found no acceptable evidence.
    if not retrieved_documents:

        answer = (
            "The available documents do not contain "
            "enough relevant information to answer "
            "this question."
        )

    else:

        # Stage 4: include retrieved evidence and history in a constrained prompt.
        prompt = build_rag_prompt(
            question=question,
            retrieved_documents=retrieved_documents,
            chat_history=chat_history
        )

        # Stage 5: ask Gemini to formulate the response from that prompt.
        answer = generate_response(prompt)

    # Return the original wording, rewritten search query, answer, and citations.
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
    # Retrieve evidence across PDFs and synthesize it into one answer.
    # Search across all sources so the synthesis can combine document evidence.
    retrieved_documents = search_context(
        index=index,
        documents=documents,
        question=question,
        top_k=top_k,
        min_similarity=min_similarity
    )

    # Return an explicit no-evidence response instead of sending empty context.
    if not retrieved_documents:
        return {
            "question": question,
            "answer": "I couldn't find enough relevant information in your uploaded documents.",
            "sources": []
        }

    # The LLM helper formats citations and asks for cross-source synthesis.
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
    # Check whether the knowledge base contains supporting evidence.
    # Gather likely evidence; the count is a simple coverage signal, not proof.
    retrieved_documents = search_context(
        index=index,
        documents=documents,
        question=question,
        top_k=top_k,
        min_similarity=min_similarity
    )

    # List distinct PDFs that contributed at least one retrieved passage.
    unique_sources = sorted({
        document["source"]
        for document in retrieved_documents
    })

    # Separate no evidence, limited evidence, and a larger evidence set for UI use.
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

    # Include the passages themselves so callers can inspect the basis for status.
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
    # Compare relevant information from two PDF documents.
    # Validate both requested sources before spending time on retrieval.
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

    # Retrieve each side independently so matches from one PDF cannot crowd out
    # matches from the other PDF in the global top-k results.
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

    # Ask the comparison prompt to describe similarities and differences grounded
    # in the separately retrieved contexts.
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
    # Discover related concepts from uploaded documents.
    # Retrieve evidence first; concept generation is grounded in these passages.
    retrieved_documents = search_context(
        index=index,
        documents=documents,
        question=question,
        top_k=top_k,
        min_similarity=min_similarity
    )

    # Avoid an LLM request when the knowledge base offers no matching context.
    if not retrieved_documents:
        return {
            "question": question,
            "concepts": "No relevant concepts found.",
            "sources": []
        }

    # Format and analyze the retrieved passages with the discovery-specific prompt.
    concepts = discover_related_concepts(
        question,
        retrieved_documents
    )

    return {
        "question": question,
        "concepts": concepts,
        "sources": retrieved_documents
    }
