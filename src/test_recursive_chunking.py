
from chunking.text_chunker import recursive_chunk_text

sample_text = """
Retrieval-Augmented Generation combines information retrieval
with large language models.

RAG retrieves relevant information from external knowledge
sources and provides that information to an LLM.

Embeddings are numerical representations of text.
They help computers identify semantic similarities.

FAISS is a library used for efficient similarity search
and clustering of dense vectors.
"""

chunks = recursive_chunk_text(
    sample_text,
    chunk_size=100,
    overlap=20
)

print("TOTAL CHUNKS:", len(chunks))

for index, chunk in enumerate(chunks):

    print("\n" + "=" * 50)

    print("CHUNK NUMBER:", index + 1)

    print("CHUNK LENGTH:", len(chunk))

    print("\nCHUNK CONTENT:")

    print(chunk)
