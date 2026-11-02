import re
import unicodedata


def clean_text(text):

    # Step 1: Validate input
    if not isinstance(text, str):
        return ""

    # Step 2: Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Step 3: Remove unwanted control characters
    text = "".join(
        char for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )

    # Step 4: Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Step 5: Remove unnecessary spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Step 6: Remove spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)

    # Step 7: Replace excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 8: Remove leading and trailing spaces
    text = text.strip()

    return text