
from pathlib import Path
import pymupdf


def extract_pdf(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    documents = []

    with pymupdf.open(file_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text", sort=True)

            if not text.strip():
                continue

            document = {
                "source": file_path.name,
                "file_type": "pdf",
                "page_number": page_number,
                "text": text
            }

            documents.append(document)

    if not documents:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be scanned or image-based."
        )

    return documents
