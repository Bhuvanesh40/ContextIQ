import re
import unicodedata


def clean_text(text):
    # Normalize extracted document text before chunking and embedding.
    # Step 1: Validate input
    if not isinstance(text, str):
        return ""

    # Normalize equivalent Unicode forms (for example, compatibility glyphs).
    text = unicodedata.normalize("NFKC", text)

    # Drop non-printing controls while preserving tabs and line breaks.
    text = "".join(
        char for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )

    # Convert Windows and classic Mac line endings to the Unix newline form.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse runs of horizontal whitespace without joining separate lines.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove indentation/trailing spaces left around each normalized newline.
    text = re.sub(r" *\n *", "\n", text)

    # Keep at most one empty line so paragraphs remain visually distinct.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 8: Remove leading and trailing spaces
    text = text.strip()

    return text