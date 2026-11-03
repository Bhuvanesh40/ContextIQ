
# Recursive text splitting prefers paragraph and sentence boundaries.


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

    # Validate settings up front so recursive splitting always makes progress.
    if not isinstance(text, str):
        raise TypeError("Text must be a string")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    # Strip only outer whitespace before checking for empty input.
    text = text.strip()

    if not text:
        return []

    # Prefer semantic boundaries, falling back to spaces and individual characters.
    separators = ["\n\n", "\n", ". ", " ", ""]

    # Split oversized passages using progressively finer separators.
    def split_recursively(text, separators):

        # Do not subdivide a passage already within the configured maximum.
        if len(text) <= chunk_size:
            return [text]

        if not separators:
            return [
                text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]

        separator = separators[0]
        remaining_separators = separators[1:]

        # Character-level slicing guarantees a bounded result for unbroken text.
        if separator == "":
            return [
                text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]

        # Try the next separator if this text has no instance of the current one.
        if separator not in text:
            return split_recursively(text, remaining_separators)

        # Restore separators after splitting so punctuation/newline context is kept.
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

    # Convert source text into atomic pieces small enough to pack into chunks.
    pieces = split_recursively(text, separators)

    # Greedily combine adjacent pieces without exceeding the maximum chunk size.
    chunks = []
    current_chunk = ""

    for piece in pieces:

        if len(current_chunk) + len(piece) <= chunk_size:
            current_chunk += piece

        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Repeat a suffix of the prior chunk so adjacent chunks share context.
            prefix = current_chunk[-overlap:] if overlap else ""

            if len(prefix) + len(piece) <= chunk_size:
                current_chunk = prefix + piece
            else:
                current_chunk = piece

    # Include the last partially filled chunk after the loop finishes.
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
