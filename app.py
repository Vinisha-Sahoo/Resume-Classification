import streamlit as st
import joblib
import os
import re
from docx import Document
from collections import Counter

# === Absolute path setup ===
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")

# === Model list with accuracy ===
MODELS = {
    "Logistic Regression": ("logistic_model.pkl", 0.9367),
    "Random Forest": ("randomforest_model.pkl", 0.942),
    "Naive Bayes": ("naivebayes_model.pkl", 0.918),
    "SVM": ("svm_model.pkl", 0.934),
    "KNN": ("knn_model.pkl", 0.91)
}

# === Role Info ===
role_details = {
    "Workday Consultant": {
        "Keywords": ["Workday", "EIB", "Studio", "XSLT", "PICOF", "PECI"],
        "Description": "Specialist in Workday integrations and automation tools."
    },
    "PeopleSoft Consultant": {
        "Keywords": ["PeopleSoft", "FSCM", "Application Engine", "Component Interface"],
        "Description": "Expert in Oracle PeopleSoft ERP modules."
    },
    "SQL Developer": {
        "Keywords": ["T-SQL", "SQL Server", "Stored Procedures", "SSIS", "ETL"],
        "Description": "Database development and query optimization expert."
    },
    "Frontend Developer": {
        "Keywords": ["React", "JavaScript", "HTML", "CSS", "Redux"],
        "Description": "Creates dynamic user interfaces using modern web technologies."
    },
    "ETL Developer": {
        "Keywords": ["Informatica", "ETL", "SSRS", "Data Warehouse"],
        "Description": "Builds and manages large-scale ETL data pipelines."
    }
}

# === Utilities ===
def load_pickle(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        st.error(f"❌ Missing: `{filename}` inside the models/ folder.")
        st.stop()
    return joblib.load(path)

def extract_text_from_docx(file):
    doc = Document(file)
    return ' '.join([para.text for para in doc.paragraphs])

def extract_sections(text):
    text = text.lower()
    exp = re.search(r"(experience|work history)(.*?)(education|skills|projects|$)", text, re.DOTALL)
    role = re.search(r"(roles|responsibilities)(.*?)(experience|education|skills|projects|$)", text, re.DOTALL)
    return (
        exp.group(2).strip() if exp else "Not found",
        role.group(2).strip() if role else "Not found"
    )

def extract_keywords(text, top_n=10):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = text.lower().split()
    common_words = set(["and", "the", "with", "for", "from", "that", "this", "have", "has", "was", "are", "will", "shall", "our", "but", "you", "your"])
    filtered = [w for w in words if len(w) > 3 and w not in common_words]
    return [kw for kw, _ in Counter(filtered).most_common(top_n)]

# === Streamlit UI ===
st.set_page_config("Resume Classifier", layout="wide")
st.title("📄 Resume Role Classifier")
st.markdown("Upload your resume and let AI predict your job category using different ML models.")

# === Sidebar ===
st.sidebar.header("🔧 Choose a Model")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
model_file, accuracy = MODELS[model_name]
st.sidebar.markdown(f"**Accuracy:** `{accuracy * 100:.2f}%`")

# === File Upload ===
uploaded_file = st.file_uploader("📤 Upload a `.docx` resume", type=["docx"])

if uploaded_file:
    st.info("⏳ Analyzing resume...")

    model = load_pickle(model_file)
    vectorizer = load_pickle("tfidf_vectorizer.pkl")
    label_encoder = load_pickle("label_encoder.pkl")

    raw_text = extract_text_from_docx(uploaded_file)
    X = vectorizer.transform([raw_text])
    prediction = model.predict(X)[0]
    predicted_role = label_encoder.inverse_transform([prediction])[0]

    # 🎯 Prediction
    st.success(f"🎯 **Predicted Role:** {predicted_role}")

    with st.expander("💼 Role Details", expanded=True):
        if predicted_role in role_details:
            role = role_details[predicted_role]
            st.markdown(f"**📝 Description:** {role['Description']}")
            st.markdown("**📌 Keywords:**")
            st.markdown("\n".join(f"- {kw}" for kw in role["Keywords"]))
        else:
            st.markdown("**📝 Description:** _(No predefined role description)_")
            st.markdown("**📌 Extracted Keywords from Resume:**")
            for kw in extract_keywords(raw_text):
                st.markdown(f"- {kw}")

    # 🧾 Experience & Responsibilities
    exp_text, role_text = extract_sections(raw_text)
    st.subheader("📚 Experience")
    st.markdown(exp_text[:1000] + "..." if exp_text != "Not found" else "Not found")

    st.subheader("🧰 Responsibilities")
    st.markdown(role_text[:1000] + "..." if role_text != "Not found" else "Not found")

else:
    st.warning("📎 Please upload a `.docx` file to begin.")
