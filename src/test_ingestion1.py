from ingestion.document_loader import extract_pdf


file_path = "data/uploads/sample.pdf"

documents = extract_pdf(file_path)

print("Total extracted pages:", len(documents))

for document in documents:

    print("\nSOURCE:", document["source"])

    print("PAGE:", document["page_number"])

    print("\nCONTENT:")

    print(document["text"][:500])