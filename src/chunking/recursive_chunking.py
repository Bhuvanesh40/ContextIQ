
def recursive_chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into meaningful chunks using recursive splitting.

    Args:
        text: Input text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of text chunks.
    """

    # Step 1: Validate input
    if not isinstance(text, str):
        raise TypeError("Text must be a string")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    # Step 2: Remove leading and trailing spaces
    text = text.strip()

    if not text:
        return []

    # Step 3: Define splitting priority
    separators = ["\n\n", "\n", ". ", " ", ""]

    # Step 4: Recursively split the text
    def split_recursively(text, separators):

        if len(text) <= chunk_size:
            return [text]

        if not separators:
            return [
                text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]

        separator = separators[0]
        remaining_separators = separators[1:]

        # Character-level fallback
        if separator == "":
            return [
                text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]

        if separator not in text:
            return split_recursively(text, remaining_separators)

        # Split while preserving separators
        parts = text.split(separator)
        pieces = []

        for index, part in enumerate(parts):

            if not part:
                continue

            # Preserve the separator after each part
            if index < len(parts) - 1:
                part += separator

            if len(part) <= chunk_size:
                pieces.append(part)
            else:
                pieces.extend(
                    split_recursively(part, remaining_separators)
                )

        return pieces

    pieces = split_recursively(text, separators)

    # Step 5: Combine smaller pieces into chunks
    chunks = []
    current_chunk = ""

    for piece in pieces:

        if len(current_chunk) + len(piece) <= chunk_size:
            current_chunk += piece

        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Step 6: Preserve context using overlap
            prefix = current_chunk[-overlap:] if overlap else ""

            if len(prefix) + len(piece) <= chunk_size:
                current_chunk = prefix + piece
            else:
                current_chunk = piece

    # Step 7: Store the final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
