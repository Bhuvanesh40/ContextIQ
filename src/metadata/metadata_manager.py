
def attach_metadata(chunks, source, page_number=None, start_index=0):
    # Wrap text chunks with stable IDs and their document/page provenance.
    # Build records in input order because FAISS uses that same order for vector IDs.
    documents = []

    for index, chunk in enumerate(chunks):
        # Offset IDs when appending pages/documents so IDs remain unique.
        document = {
            "chunk_id": f"chunk_{index + start_index + 1:04d}",
            "source": source,
            "page_number": page_number,
            "chunk_text": chunk
        }

        # Keep each chunk's source and page alongside its text for citations.
        documents.append(document)

    return documents
