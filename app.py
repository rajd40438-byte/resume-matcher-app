"""
Streamlit app: Resume Category Classifier + Resume-to-Job-Description Matcher.
Run locally with: streamlit run app.py
Deploy free at: https://share.streamlit.io

Uses:
  - A TF-IDF + trained classifier (from eda_and_modeling.py) to predict
    the resume's job category.
  - Sentence embeddings (sentence-transformers) to compute a semantic
    match score between a resume and a specific job description —
    this catches meaning-level overlap, not just exact keyword overlap
    (e.g. "led a team" vs "managed people" score as similar).
  - A curated skill list to show which JD-mentioned skills are present
    or missing in the resume — explainable and easy to extend.
"""

import re
import streamlit as st
import joblib
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from skills_list import SKILLS

st.set_page_config(page_title="Resume-JD Matcher", layout="wide")

@st.cache_resource
def load_artifacts():
    classifier = joblib.load("outputs/resume_classifier.pkl")
    vectorizer = joblib.load("outputs/tfidf_vectorizer.pkl")
    # WHY this specific model: all-MiniLM-L6-v2 is small (~80MB), fast
    # enough for a free-tier deployment, and a standard, well-benchmarked
    # choice for semantic similarity tasks.
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return classifier, vectorizer, embedder

classifier, vectorizer, embedder = load_artifacts()

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def extract_skills(text):
    text_lower = text.lower()
    return sorted({skill for skill in SKILLS if skill in text_lower})

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

st.title("📄 Resume-Job Description Matcher")
st.write(
    "Paste your resume and a target job description to get a match "
    "score, missing-skills gap analysis, and a predicted resume category."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Resume")
    resume_input_mode = st.radio("Input method", ["Upload PDF", "Paste text"], horizontal=True, key="resume_mode")
    resume_text = ""
    if resume_input_mode == "Upload PDF":
        resume_pdf = st.file_uploader("Upload resume PDF", type="pdf", key="resume_pdf")
        if resume_pdf is not None:
            resume_text = extract_text_from_pdf(resume_pdf)
            with st.expander("Preview extracted text"):
                st.text(resume_text[:2000])
    else:
        resume_text = st.text_area("Paste resume text", height=300, key="resume_paste")

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area("Paste the job description here...", height=380, key="jd_paste")

if st.button("Analyze Match"):
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please paste both your resume and a job description.")
    else:
        # --- Semantic match score ---
        embeddings = embedder.encode([resume_text, jd_text], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        match_pct = round(similarity * 100, 1)

        st.metric("Semantic Match Score", f"{match_pct}%")
        if match_pct >= 70:
            st.success("Strong match — your resume's content closely aligns with this JD.")
        elif match_pct >= 45:
            st.info("Moderate match — consider tailoring your resume further for this role.")
        else:
            st.warning("Low match — this resume may need significant tailoring for this JD.")

        # --- Skill gap analysis ---
        resume_skills = set(extract_skills(resume_text))
        jd_skills = set(extract_skills(jd_text))
        missing = sorted(jd_skills - resume_skills)
        matched = sorted(jd_skills & resume_skills)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("✅ Matched skills")
            st.write(", ".join(matched) if matched else "None detected from the curated skill list.")
        with col4:
            st.subheader("⚠️ Skills in JD but missing from your resume")
            st.write(", ".join(missing) if missing else "No gaps detected — nice.")

        # --- Predicted resume category ---
        cleaned = clean_text(resume_text)
        X = vectorizer.transform([cleaned])
        predicted_category = classifier.predict(X)[0]
        st.subheader("Predicted resume category")
        st.write(f"The classifier trained on labeled resumes predicts this resume best fits: **{predicted_category}**")

st.divider()
st.caption(
    "Match score: sentence-embedding cosine similarity (all-MiniLM-L6-v2). "
    "Skill gaps: curated keyword list (see skills_list.py) — extend it for your target roles. "
    "Category: TF-IDF + classifier trained on a labeled resume dataset."
)
