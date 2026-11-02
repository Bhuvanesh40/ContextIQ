
def build_rag_prompt(
    question,
    retrieved_documents,
    chat_history=None
):

    if not retrieved_documents:
        raise ValueError("No relevant documents were retrieved.")

    context_parts = []

    for index, document in enumerate(retrieved_documents, start=1):

        context_parts.append(
            f"""
[Source {index}]
Document: {document['source']}
Page: {document['page_number']}

Content:
{document['chunk_text']}
"""
        )

    context = "\n".join(context_parts)

    history_text = "No previous conversation."

    if chat_history:
        history_parts = []

        for message in chat_history[-6:]:
            history_parts.append(
                f"{message['role'].upper()}: {message['content']}"
            )

        history_text = "\n".join(history_parts)

    prompt = f"""
You are ContextIQ, an AI knowledge assistant.

PREVIOUS CONVERSATION:
{history_text}

CURRENT USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:

1. Answer using the retrieved document context only.
2. Use conversation history to understand the user's intent.
3. Do not treat previous assistant answers as factual evidence.
4. Use the current retrieved documents as the factual source.
5. Explain clearly and naturally.
6. Do not invent unsupported information.
7. Cite relevant claims using:
   (Source: filename, Page: page_number).
8. If the retrieved context does not contain enough
   information, clearly say so.

ANSWER:
"""

    return prompt
