
from ingestion.document_loader import extract_pdf

pdf_path = "data/uploads/sample.pdf"

documents = extract_pdf(pdf_path)

print("PDF EXTRACTION SUCCESSFUL")
print("TOTAL PAGES EXTRACTED:", len(documents))

for document in documents:

    print("\n" + "=" * 50)

    print("SOURCE:", document["source"])
    print("PAGE:", document["page_number"])

    print("\nEXTRACTED TEXT:")
    print(document["text"][:500])
