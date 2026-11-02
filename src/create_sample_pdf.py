from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

output_folder = Path(__file__).resolve().parent.parent / "data" / "uploads"
output_folder.mkdir(parents=True, exist_ok=True)


def create_pdf(filename, title, lines):
    pdf_path = output_folder / filename
    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 800, title)
    pdf.setFont("Helvetica", 12)

    y = 750
    for line in lines:
        if y < 60:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y = 800
        pdf.drawString(50, y, line)
        y -= 25

    pdf.save()
    print(f"Created {pdf_path.name} at {pdf_path}")


def main():
    create_pdf(
        "sample.pdf",
        "ContextIQ - RAG Knowledge Document",
        [
            "Retrieval-Augmented Generation (RAG)",
            "RAG combines information retrieval with large language models.",
            "It retrieves relevant information from external knowledge sources.",
            "The LLM uses retrieved context to generate grounded answers.",
            "Embeddings convert text into numerical vectors.",
            "FAISS is a library for efficient vector similarity search.",
            "Text chunking divides large documents into smaller sections."
        ]
    )

    create_pdf(
        "python_guide.pdf",
        "Python Programming Guide",
        [
            "Python Overview",
            "Python is a popular programming language.",
            "It emphasizes readability and rapid development.",
            "Python is widely used in Machine Learning, AI Engineering, and Web Development."
        ]
    )

    create_pdf(
        "llm_guide.pdf",
        "Large Language Models Guide",
        [
            "Large Language Models Overview",
            "Large language models generate natural language responses.",
            "Transformers are the foundational architecture for modern LLMs.",
            "Grounding LLMs with external context prevents hallucination."
        ]
    )


if __name__ == "__main__":
    main()
