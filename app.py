from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# Load DOCX
docx_path = "input.docx"
doc = Document(docx_path)

# Extract all text
full_text = []
for para in doc.paragraphs:
    if para.text.strip():
        full_text.append(para.text.strip())

# 🛠 Here you can keep your formatting logic
formatted_text = "\n".join(full_text)  # Replace with your bullet-point logic

# Save to PDF
pdf_path = "output.pdf"
styles = getSampleStyleSheet()
story = []
for line in formatted_text.split("\n"):
    story.append(Paragraph(line, styles["Normal"]))
    story.append(Spacer(1, 8))

pdf = SimpleDocTemplate(pdf_path, pagesize=letter)
pdf.build(story)

print(f"PDF saved as {pdf_path}")
