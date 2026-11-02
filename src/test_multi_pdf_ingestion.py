from pathlib import Path

from ingestion.knowledge_base_manager import ingest_pdf


def main():
    project_root = Path(__file__).resolve().parent.parent

    pdf_input = input("Enter PDF path: ").strip().strip('"')
    pdf_path = Path(pdf_input)

    # Resolve relative paths from the project root.
    if not pdf_path.is_absolute():
        pdf_path = project_root / pdf_path

    if not pdf_path.exists():
        print(f"\nERROR: PDF not found: {pdf_path}")
        print("\nPDF files available in data/uploads:")

        upload_folder = project_root / "data" / "uploads"

        if upload_folder.exists():
            for file in upload_folder.glob("*.pdf"):
                print(f"- {file.name}")

        return

    result = ingest_pdf(pdf_path)

    print("\nSTATUS:", result["status"])
    print("MESSAGE:", result["message"])
    print("TOTAL VECTORS:", result["index"].ntotal)
    print("TOTAL METADATA RECORDS:", len(result["documents"]))


if __name__ == "__main__":
    main()