from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Place generated examples beside the other documents used by the app.
output_folder = Path(__file__).resolve().parent.parent / "data" / "uploads"
output_folder.mkdir(parents=True, exist_ok=True)


def create_pdf(filename, title, lines):
    # Render a title and plain text lines into a simple example PDF.
    # Build the target path and open a ReportLab canvas using A4 page dimensions.
    pdf_path = output_folder / filename
    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)

    # Draw the title with stronger typography, then switch to body text.
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 800, title)
    pdf.setFont("Helvetica", 12)

    y = 750
    for line in lines:
        # Start a fresh page when the next line would run below the page margin.
        if y < 60:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y = 800
        # Draw this line at the left margin, then move down for the next line.
        pdf.drawString(50, y, line)
        y -= 25

    # Finalize the PDF file and report where it was written.
    pdf.save()
    print(f"Created {pdf_path.name} at {pdf_path}")


def main():
    # Generate the three small PDFs used for demos and local evaluation.
    # Keep each sample's content in a list of printable lines for create_pdf().
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
