def chunk_text(text, chunk_size=500, overlap=100):

    # Step 1: Validate input
    if not isinstance(text, str):
        raise TypeError("Text must be a string")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    # Step 2: Remove leading and trailing spaces
    text = text.strip()

    if not text:
        return []

    chunks = []

    # Step 3: Initialize starting position
    start = 0

    # Step 4: Divide text into chunks
    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        # Step 5: Store the chunk
        if chunk.strip():
            chunks.append(chunk.strip())

        # Step 6: Move forward with overlap
        start += chunk_size - overlap

    return chunks


def recursive_chunk_text(text, chunk_size=500, overlap=50):

    if not isinstance(text, str):
        raise TypeError("Text must be a string")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    text = text.strip()

    if not text:
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_recursively(text, separators):

        if len(text) <= chunk_size:
            return [text]

        if not separators:
            return [text[i:i + chunk_size]
                    for i in range(0, len(text), chunk_size)]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return [
                text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]

        if separator not in text:
            return split_recursively(text, remaining_separators)

        parts = text.split(separator)

        chunks = []

        for index, part in enumerate(parts):

            if not part.strip():
                continue

            # Restore the separator between pieces
            if index < len(parts) - 1:
                part += separator

            if len(part) <= chunk_size:
                chunks.append(part)
            else:
                chunks.extend(
                    split_recursively(part, remaining_separators)
                )

        return chunks

    # First, split the text using meaningful boundaries
    pieces = split_recursively(text, separators)

    # Combine smaller pieces without exceeding chunk_size
    chunks = []
    current_chunk = ""

    for piece in pieces:

        if len(current_chunk) + len(piece) <= chunk_size:
            current_chunk += piece

        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Preserve some context from the previous chunk
            prefix = current_chunk[-overlap:] if overlap else ""

            # If the prefix leaves insufficient space, omit it
            if len(prefix) + len(piece) > chunk_size:
                prefix = ""

            current_chunk = prefix + piece

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
