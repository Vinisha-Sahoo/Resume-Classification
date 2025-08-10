import re
import streamlit as st
from PyPDF2 import PdfReader

# --- Helper to extract sections ---
def extract_sections(text):
    skill_pattern = r"(?i)(?:skillsets?|skills)\s*[:\-]?\s*(.*?)(?=(?:responsibilit|experience|$))"
    resp_pattern = r"(?i)(?:responsibilit(?:y|ies)|roles?)\s*[:\-]?\s*(.*?)(?=(?:skill|experience|$))"

    skill_match = re.search(skill_pattern, text, re.S)
    resp_match = re.search(resp_pattern, text, re.S)

    skill_text = skill_match.group(1).strip() if skill_match else "Not found"
    resp_text = resp_match.group(1).strip() if resp_match else "Not found"

    return skill_text, resp_text

# --- Group lines that belong together ---
def group_related_points(text):
    lines = [line.strip() for line in re.split(r'[\n•]', text) if line.strip()]
    grouped = []
    buffer = []

    for line in lines:
        if re.match(r"(?i).*\b(experience|project|worked|developed|managed|responsible)\b", line) and buffer:
            grouped.append(" ".join(buffer))
            buffer = [line]
        else:
            buffer.append(line)

    if buffer:
        grouped.append(" ".join(buffer))

    return "\n".join(f"- {point}" for point in grouped)

# --- Streamlit UI ---
st.title("📄 Resume Parser")

uploaded_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    raw_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    exp_text, role_text = extract_sections(raw_text)

    st.write("---")

    st.subheader("🛠 Skillsets")
    if exp_text != "Not found":
        st.markdown(group_related_points(exp_text[:1500]) + "...")
    else:
        st.markdown("Not found")

    st.subheader("🧰 Responsibilities")
    if role_text != "Not found":
        st.markdown(group_related_points(role_text[:1500]) + "...")
    else:
        st.markdown("Not found")
