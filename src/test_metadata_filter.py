
from rag_pipeline import load_knowledge_base, search_context

# Load saved knowledge base
index, documents = load_knowledge_base()

question = "What is Retrieval-Augmented Generation?"

print("\nTEST 1: SEARCH ALL DOCUMENTS")

all_results = search_context(
    index=index,
    documents=documents,
    question=question,
    top_k=3,
    min_similarity=0.0
)

print("\nTEST 2: SEARCH ONLY sample.pdf")

source_results = search_context(
    index=index,
    documents=documents,
    question=question,
    top_k=3,
    min_similarity=0.0,
    source_filter="sample.pdf"
)

print("\nTEST 3: SEARCH ONLY PAGE 1")

page_results = search_context(
    index=index,
    documents=documents,
    question=question,
    top_k=3,
    min_similarity=0.0,
    page_filter=1
)

# Verify filtering
assert all(
    result["source"] == "sample.pdf"
    for result in source_results
)

assert all(
    result["page_number"] == 1
    for result in page_results
)

print("\nALL METADATA FILTER TESTS PASSED!")
