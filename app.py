import streamlit as st
import fitz
import re
from pdf2image import convert_from_path
from PIL import Image
import tempfile, os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

poppler_path = r"C:\New folder\poppler-25.11.0\Library\bin"

st.set_page_config(page_title="Resume/CV Parser", layout="centered")
st.title("Resume/CV Parser")

@st.cache_resource
def load_ner():
    model_id = "Nabinp10/resume-ner-output"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

ner = load_ner()

# ----------------- PDF / Image extraction -----------------
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ----------------- Name extraction -----------------
def extract_name(entities):
    name = ""
    for e in entities:
        if e["entity_group"] in ["PER", "PERSON", "NAME"]:
            word = e["word"].replace("##", "").strip()
            name += " " + word
    return name.strip() or "Not found"

# ----------------- Extract entities from text -----------------
def extract_entities(text):
    entities = ner(text)
    name = location = ""

    for ent in entities:
        if ent["entity_group"] == "PER" and not name:
            name = ent["word"].replace("##", " ").strip()
        if ent["entity_group"] == "LOC" and not location:
            location = ent["word"].replace("##", " ").strip()

    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.search(r"[\+]?[0-9]{8,15}", text.replace(" ", "").replace("-", ""))

    skills_list = [
        "Python", "Java", "JavaScript", "React", "Node.js", "Django", "Flask", "AWS", "Docker",
        "Machine Learning", "NLP", "Deep Learning", "PyTorch", "TensorFlow", "SQL", "MongoDB",
        "Git", "Linux", "HTML", "CSS", "FastAPI", "LangChain", "Hugging Face", "Streamlit"
    ]
    skills_found = [s for s in skills_list if s.lower() in text.lower()]

    education_keywords = ["BSc", "MSc", "Bachelor", "Master", "B.E.", "B.Tech", "Diploma", "PhD"]
    education_lines = [line.strip() for line in text.split("\n")
                       if any(kw in line for kw in education_keywords) or "University" in line or "College" in line]
    education = " | ".join(education_lines[:2]) if education_lines else "Not found"

    experience_lines = [line.strip() for line in text.split("\n")
                        if re.search(r"\d{4}.*\d{4}|Present|Current", line)
                        and any(x in line for x in ["Engineer", "Intern", "Developer", "Analyst"])]
    experience = " | ".join(experience_lines[:3]) if experience_lines else "Not found"

    return {
        "Name": name or "Not found",
        "Email": email.group() if email else "Not found",
        "Phone": phone.group() if phone else "Not found",
        "Location": location or "Not found",
        "Skills": ", ".join(skills_found) if skills_found else "Not found",
        "Education": education,
        "Experience": experience
    }

# ----------------- MAIN APP -----------------
uploaded_file = st.file_uploader("Upload Resume (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # PREVIEW
    if uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        images = convert_from_path(tmp_path, first_page=1, last_page=1, poppler_path=poppler_path)
        st.image(images[0], caption="Resume Preview", width=600)
        os.unlink(tmp_path)
    else:
        st.image(uploaded_file, caption="Image Preview", width=600)

    # Extract text
    with st.spinner("Reading resume text..."):
        text = extract_text_from_pdf(uploaded_file.getvalue())

    # Extract entities
    with st.spinner("Extracting information..."):
        result = extract_entities(text)
        raw_entities = ner(text)

    # DISPLAY RESULTS
    st.subheader("Extracted Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Name", extract_name(raw_entities))
        st.write(f"**Email**: {result['Email']}")
        st.write(f"**Phone**: {result['Phone']}")
        st.write(f"**Location**: {result['Location']}")
    with col2:
        st.write(f"**Skills**: {result['Skills']}")
        st.write(f"**Education**: {result['Education']}")
        st.write(f"**Experience**: {result['Experience']}")

    st.json(result)
    st.success("✅ Resume parsing complete!")

else:
    st.info("Please upload a resume to get started")
