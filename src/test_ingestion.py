from ingestion.document_loader import extract_txt

# Step 1: Specify the text file
file_path = "data/uploads/sample.txt"

# Step 2: Extract the document
documents = extract_txt(file_path)

# Step 3: Display the results
print("Total extracted documents:", len(documents))

for document in documents:

    print("\nSOURCE:", document["source"])

    print("FILE TYPE:", document["file_type"])

    print("\nEXTRACTED CONTENT:")

    print(document["text"])