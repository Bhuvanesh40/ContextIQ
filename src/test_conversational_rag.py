
from rag_pipeline import (
    load_knowledge_base,
    generate_rag_answer
)


# Load saved knowledge base
print("\nLOADING CONTEXTIQ KNOWLEDGE BASE")

index, documents = load_knowledge_base()

# Store conversation messages
chat_history = []


def ask_contextiq(question):

    print("\n" + "=" * 60)
    print("USER:", question)

    result = generate_rag_answer(
        index=index,
        documents=documents,
        question=question,
        top_k=3,
        min_similarity=0.20,
        chat_history=chat_history
    )

    print("\nCONTEXTIQ:")
    print(result["answer"])

    print("\nSOURCES:")

    for source in result["sources"]:
        print(
            f"- {source['source']} | "
            f"Page: {source['page_number']}"
        )

    # Save the original user question and generated answer
    chat_history.append({
        "role": "user",
        "content": question
    })

    chat_history.append({
        "role": "assistant",
        "content": result["answer"]
    })


# First question
ask_contextiq(
    "What is Retrieval-Augmented Generation?"
)

# Follow-up question
ask_contextiq(
    "What does it combine?"
)

# Another follow-up
ask_contextiq(
    "Why are large language models useful in it?"
)
