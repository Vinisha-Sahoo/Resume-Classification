from docx import Document
import re

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())
    return "\n".join(text)

def group_related_points(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    grouped_points = []
    current_point = ""

    for line in lines:
        # New bullet point starts if:
        # - line begins with a bullet/number OR
        # - line starts with uppercase and current_point isn't empty
        if re.match(r"^(\d+[\.\)]|\-|\•)", line) or (line[0].isupper() and current_point):
            grouped_points.append(current_point.strip())
            current_point = line
        else:
            current_point += " " + line

    if current_point:
        grouped_points.append(current_point.strip())

    return grouped_points

if __name__ == "__main__":
    docx_path = "input.docx"  # Change this to your file name
    raw_text = extract_text_from_docx(docx_path)
    grouped = group_related_points(raw_text)

    print("\n--- Grouped Points ---\n")
    for idx, point in enumerate(grouped, 1):
        print(f"{idx}. {point}")
