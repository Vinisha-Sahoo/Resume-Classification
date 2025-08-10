from docx import Document
from fpdf import FPDF

def read_docx(file_path):
    doc = Document(file_path)
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return lines

def group_related_lines(lines):
    grouped = []
    temp_group = []

    for line in lines:
        if line[0].isupper() and temp_group:
            grouped.append(" ".join(temp_group))
            temp_group = [line]
        else:
            temp_group.append(line)
    if temp_group:
        grouped.append(" ".join(temp_group))
    return grouped

def save_as_pdf(grouped_points, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Formatted Resume", ln=True, align="C")
    pdf.ln(10)

    for idx, point in enumerate(grouped_points, start=1):
        pdf.multi_cell(0, 8, f"{idx}. {point}")
        pdf.ln(1)

    pdf.output(output_path)

# === USAGE ===
input_docx = "resume.docx"   # Your input file
output_pdf = "formatted_resume.pdf"

lines = read_docx(input_docx)
grouped_points = group_related_lines(lines)
save_as_pdf(grouped_points, output_pdf)

print(f"Saved formatted PDF to {output_pdf}")
