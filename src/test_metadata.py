
from chunking.text_chunker import recursive_chunk_text
from metadata.metadata_manager import attach_metadata

sample_text = """
Retrieval-Augmented Generation combines information retrieval
with large language models.

RAG retrieves relevant information from external knowledge
sources and provides that information to an LLM.

Embeddings are numerical representations of text.
They help computers identify semantic similarities.
"""

# Step 1: Create chunks
chunks = recursive_chunk_text(
    sample_text,
    chunk_size=100,
    overlap=20
)

# Step 2: Attach metadata
documents = attach_metadata(
    chunks=chunks,
    source="sample.pdf",
    page_number=1
)

# Step 3: Display metadata
print("TOTAL DOCUMENT CHUNKS:", len(documents))

for document in documents:

    print("\n" + "=" * 50)

    print("CHUNK ID:", document["chunk_id"])

    print("SOURCE:", document["source"])

    print("PAGE NUMBER:", document["page_number"])

    print("CHUNK TEXT:")

    print(document["chunk_text"])
