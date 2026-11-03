# Character-based chunking helpers split long text for retrieval.


def chunk_text(text, chunk_size=500, overlap=100):

    # Reject invalid sizes before calculating a step that could be zero/negative.
    if not isinstance(text, str):
        raise TypeError("Text must be a string")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    # Trim only the outer whitespace; internal text remains available to split.
    text = text.strip()

    if not text:
        return []

    chunks = []

    # `start` marks the first character included in the next window.
    start = 0

    # Advance by window size minus overlap to retain context at each boundary.
    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        # Ignore whitespace-only windows so they do not become search documents.
        if chunk.strip():
            chunks.append(chunk.strip())

        # The next window begins before this one ended by the requested overlap.
        start += chunk_size - overlap

    return chunks


def recursive_chunk_text(text, chunk_size=500, overlap=50):
    # Prefer natural text boundaries, then combine pieces into bounded chunks.
    # Validate settings before attempting recursive splitting.
    if not isinstance(text, str):
        raise TypeError("Text must be a string")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    # Remove outer whitespace and return early for input with no usable content.
    text = text.strip()

    if not text:
        return []

    # Try paragraph, line, sentence, word, then character boundaries in order.
    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_recursively(text, separators):
        # A piece already within the limit can be returned without more splitting.
        if len(text) <= chunk_size:
            return [text]

        if not separators:
            return [text[i:i + chunk_size]
                    for i in range(0, len(text), chunk_size)]

        # If this boundary is absent, try the next less-preferred boundary.
        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return [
                text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]

        if separator not in text:
            return split_recursively(text, remaining_separators)

        # Split on the selected boundary; it is restored below where appropriate.
        parts = text.split(separator)

        chunks = []

        for index, part in enumerate(parts):

            if not part.strip():
                continue

            # Restore the separator between pieces
            if index < len(parts) - 1:
                part += separator

            # Keep short pieces intact and subdivide oversized ones recursively.
            if len(part) <= chunk_size:
                chunks.append(part)
            else:
                chunks.extend(
                    split_recursively(part, remaining_separators)
                )

        return chunks

    # Produce small pieces before packing them into the final size-limited windows.
    pieces = split_recursively(text, separators)

    # Combine adjacent pieces until adding another would exceed the size limit.
    chunks = []
    current_chunk = ""

    for piece in pieces:

        if len(current_chunk) + len(piece) <= chunk_size:
            current_chunk += piece

        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Repeat a suffix of the previous chunk to preserve boundary context.
            prefix = current_chunk[-overlap:] if overlap else ""

            # Keep the new piece intact when the overlap would make it too large.
            if len(prefix) + len(piece) > chunk_size:
                prefix = ""

            current_chunk = prefix + piece

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
