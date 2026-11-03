
from pathlib import Path
import pymupdf


def extract_pdf(file_path):
    # Extract non-empty, page-level text records from a PDF file.
    # Normalize strings and Path inputs to one filesystem representation.
    file_path = Path(file_path)

    # Fail early with a clear error for missing files or unsupported formats.
    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    # Keep page boundaries so later chunks can cite their source page.
    documents = []

    with pymupdf.open(file_path) as pdf:
        # Sort text geometrically to reduce the chance of scrambled PDF reading order.
        for page_number, page in enumerate(pdf, start=1):

            # Extract plain text while asking PyMuPDF to sort by page layout.
            text = page.get_text("text", sort=True)

            # Scanned/image-only pages do not provide extractable text.
            if not text.strip():
                continue

            document = {
                "source": file_path.name,
                "file_type": "pdf",
                "page_number": page_number,
                "text": text
            }

            documents.append(document)

    # An empty result usually means the PDF needs OCR, which is not performed here.
    if not documents:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be scanned or image-based."
        )

    return documents
