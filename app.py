import streamlit as st
import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# -------------------------------
# EXTRACT TEXT FROM PDF
# -------------------------------
def extract_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    return text


# -------------------------------
# EXTRACT TEXT FROM DOCX
# -------------------------------
def extract_docx_text(uploaded_file):
    document = Document(uploaded_file)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + " "

    return text


# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9+#.\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text


# -------------------------------
# CALCULATE MATCH SCORE
# -------------------------------
def calculate_match(resume, job_description):

    documents = [resume, job_description]

    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    score = similarity[0][0] * 100

    return round(score, 2)


# -------------------------------
# SKILLS DATABASE
# -------------------------------
skills_database = [
    "python",
    "java",
    "c++",
    "c",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "sql",
    "mysql",
    "mongodb",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "streamlit",
    "tensorflow",
    "pytorch",
    "pandas",
    "numpy",
    "scikit-learn",
    "excel",
    "power bi",
    "tableau",
    "aws",
    "docker",
    "git",
    "github"
]


# -------------------------------
# EXTRACT SKILLS
# -------------------------------
def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in skills_database:

        if skill.lower() in text:
            found_skills.append(skill)

    return found_skills


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Powered Resume Matcher")

st.write(
    "Upload your resume and paste a job description to check how well they match."
)

st.divider()


uploaded_resume = st.file_uploader(
    "📄 Upload your Resume",
    type=["pdf", "docx"]
)


job_description = st.text_area(
    "💼 Paste the Job Description",
    height=250
)


if st.button("🔍 Analyze Resume"):

    if uploaded_resume is None:

        st.warning("Please upload your resume.")

    elif not job_description.strip():

        st.warning("Please paste a job description.")

    else:

        # Extract resume text

        if uploaded_resume.name.endswith(".pdf"):

            resume_text = extract_pdf_text(uploaded_resume)

        else:

            resume_text = extract_docx_text(uploaded_resume)


        # Clean text

        resume_clean = clean_text(resume_text)

        job_clean = clean_text(job_description)


        # Calculate score

        score = calculate_match(
            resume_clean,
            job_clean
        )


        # Extract skills

        resume_skills = extract_skills(resume_clean)

        job_skills = extract_skills(job_clean)


        matched_skills = list(
            set(resume_skills) &
            set(job_skills)
        )


        missing_skills = list(
            set(job_skills) -
            set(resume_skills)
        )


        # Display score

        st.divider()

        st.subheader("📊 Resume Match Score")

        st.progress(int(score))

        st.metric(
            "Match Percentage",
            f"{score}%"
        )


        # Analysis

        col1, col2 = st.columns(2)


        with col1:

            st.subheader("✅ Matched Skills")

            if matched_skills:

                for skill in matched_skills:

                    st.success(skill)

            else:

                st.info("No matching skills found.")


        with col2:

            st.subheader("❌ Missing Skills")

            if missing_skills:

                for skill in missing_skills:

                    st.error(skill)

            else:

                st.success("No important skills are missing!")


        # Suggestions

        st.divider()

        st.subheader("💡 AI Recommendations")


        if score >= 80:

            st.success(
                "Excellent match! Your resume is highly relevant to this job."
            )

        elif score >= 60:

            st.info(
                "Good match! Consider adding some missing skills and relevant experience."
            )

        elif score >= 40:

            st.warning(
                "Moderate match. Try improving your resume according to the job requirements."
            )

        else:

            st.error(
                "Low match. Your resume needs significant improvements for this job."
            )