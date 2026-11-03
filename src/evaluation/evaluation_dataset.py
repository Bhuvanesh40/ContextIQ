"""
ContextIQ Evaluation Ground-Truth Dataset.
Contains test queries, corresponding ground-truth chunk IDs, and reference answers.
"""

# Each record labels which stored chunk should answer a question and supplies
# a reference response for offline retrieval and generation evaluation.
evaluation_dataset = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "relevant_chunk_ids": ["chunk_0001"],
        "ground_truth": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with large language models to generate grounded responses."
    },
    {
        "question": "What are embeddings?",
        "relevant_chunk_ids": ["chunk_0002"],
        "ground_truth": "Embeddings are numerical vector representations of text that capture semantic meaning."
    },
    {
        "question": "What is Python?",
        "relevant_chunk_ids": ["chunk_0004"],
        "ground_truth": "Python is a popular programming language."
    },
    {
        "question": "What do large language models generate?",
        "relevant_chunk_ids": ["chunk_0005"],
        "ground_truth": "Large language models generate natural language responses."
    }
]
