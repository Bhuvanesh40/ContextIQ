from llm.gemini_client import generate_response


def format_context(documents):
    """Format retrieved chunks with their source information."""
    context_parts = []

    for index, document in enumerate(documents, start=1):
        context_parts.append(
            f"""
[Context {index}]
Source: {document['source']}
Page: {document['page_number']}
Similarity: {document.get('similarity_score', 'N/A')}

Content:
{document['chunk_text']}
"""
        )

    return "\n".join(context_parts)


def answer_across_documents(question, documents):
    """Generate a grounded answer using multiple retrieved documents."""
    if not documents:
        return "I couldn't find relevant information in your uploaded documents."

    context = format_context(documents)

    prompt = f"""
You are ContextIQ, an AI knowledge discovery assistant.

Answer the user's question using the provided document context.

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
1. Use only information supported by the provided context.
2. Combine information from different documents when useful.
3. Explain connections between concepts clearly.
4. Do not invent facts or unsupported relationships.
5. Cite claims using (Source: filename, Page: page_number).
6. If information is insufficient, clearly explain what is missing.
7. Avoid repeating the same information.

ANSWER:
"""
    return generate_response(prompt)


def compare_documents(question, documents_a, documents_b, source_a, source_b):
    """Compare information retrieved from two selected documents."""

    if not documents_a:
        return f"No relevant information was found in {source_a}."

    if not documents_b:
        return f"No relevant information was found in {source_b}."

    context_a = format_context(documents_a)
    context_b = format_context(documents_b)

    prompt = f"""
You are ContextIQ, an AI knowledge comparison assistant.

Compare how the two documents explain or discuss the requested topic.

QUESTION:
{question}

DOCUMENT A: {source_a}
{context_a}

DOCUMENT B: {source_b}
{context_b}

INSTRUCTIONS:
1. Explain the main points from Document A.
2. Explain the main points from Document B.
3. Identify similarities.
4. Identify differences.
5. Mention important details unique to either document.
6. Do not invent information.
7. Cite claims using (Source: filename, Page: page_number).
8. If the documents do not provide enough information for a comparison, say so.

COMPARISON:
"""
    return generate_response(prompt)


def discover_related_concepts(question, documents):
    """Identify concepts related to the question from retrieved context."""

    if not documents:
        return "No relevant concepts could be discovered from the uploaded documents."

    context = format_context(documents)

    prompt = f"""
You are ContextIQ, an AI learning assistant.

Identify concepts related to the user's question using only the
provided document context.

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
1. Identify 5 to 8 relevant concepts, if the context supports them.
2. Explain each concept in simple language.
3. Explain how each concept relates to the user's question.
4. Do not introduce concepts unsupported by the context.
5. Cite the source and page for each concept.
6. If fewer than 5 concepts are supported, provide only those.

Use this format:

1. Concept:
   Explanation:
   Connection:
   Source:

RELATED CONCEPTS:
"""
    return generate_response(prompt)