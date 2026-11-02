
def attach_metadata(chunks, source, page_number=None, start_index=0):

    documents = []

    for index, chunk in enumerate(chunks):

        document = {
            "chunk_id": f"chunk_{index + start_index + 1:04d}",
            "source": source,
            "page_number": page_number,
            "chunk_text": chunk
        }

        documents.append(document)

    return documents
