
from llm.gemini_client import generate_response


def format_chat_history(chat_history, max_messages=6):
    """
    Convert recent conversation messages into readable text.
    """
    # No history means there is no prior reference to resolve.
    if not chat_history:
        return "No previous conversation."

    # Bound history to keep rewrite prompts small and focused on recent turns.
    recent_messages = chat_history[-max_messages:]

    formatted_history = []

    # Convert structured message records to a format the language model can read.
    for message in recent_messages:
        role = message["role"].upper()
        content = message["content"]

        formatted_history.append(
            f"{role}: {content}"
        )

    return "\n".join(formatted_history)


def rewrite_question(question, chat_history):
    """
    Convert a follow-up question into a standalone question
    using previous conversation context.
    """

    # Standalone questions need no model call and can be used as-is.
    if not chat_history:
        return question

    # Include the recent turns so pronouns and short follow-ups have context.
    history = format_chat_history(chat_history)

    prompt = f"""
You are a question rewriting component in a RAG system.

Your task is to rewrite the latest user question into
a standalone question using the conversation history.

CONVERSATION HISTORY:
{history}

LATEST USER QUESTION:
{question}

RULES:
1. Resolve references such as it, they, this and that.
2. Preserve the original meaning.
3. Do not answer the question.
4. Do not add information that the user did not imply.
5. If the question is already standalone, keep its meaning.
6. Return only the rewritten question.

STANDALONE QUESTION:
"""

    # Ask the model only to rewrite the query; the RAG pipeline answers later.
    rewritten_question = generate_response(prompt).strip()

    # Keep the original question if the model unexpectedly returns no text.
    if not rewritten_question:
        return question

    return rewritten_question
