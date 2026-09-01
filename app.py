"""
app.py - AI-Powered Resume vs Job Description Matcher

An advanced Machine Learning and NLP application that analyzes resumes (PDF/DOCX)
against Job Descriptions, extracts technical skills, computes TF-IDF Cosine Similarity,
and predicts an intelligent match score using a trained Random Forest Regression model.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# 1. APPLICATION CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Resume vs Job Description Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium aesthetics
st.markdown("""
<style>
    /* Global style tweaks */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 50%, #DB2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(226, 232, 240, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 5px 0;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        color: #94A3B8;
    }
    .badge-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px 4px;
    }
    .badge-matched {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-missing {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .info-box {
        background: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
    }
    .workflow-step {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. PATHS & RESOURCE LOADERS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "regression_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

# Skills taxonomy for NLP extraction & matching
SKILLS_DATABASE = [
    "python", "java", "c++", "c#", "c", "javascript", "typescript", "go", "rust", "php",
    "ruby", "swift", "kotlin", "r", "scala", "sql", "bash", "html", "html5", "css", "css3",
    "react", "next.js", "vue", "angular", "node.js", "express", "django", "fastapi", "flask",
    "spring boot", "tailwind css", "bootstrap", "sass", "redux", "graphql", "rest api",
    "machine learning", "deep learning", "artificial intelligence", "nlp", "computer vision",
    "generative ai", "llm", "transformers", "hugging face", "langchain", "llamaindex",
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy", "scipy", "opencv",
    "tableau", "power bi", "excel", "matplotlib", "seaborn",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "gitlab", "linux", "ci/cd",
    "mysql", "postgresql", "mongodb", "redis", "sqlite", "snowflake", "bigquery", "pyspark",
    "hadoop", "kafka", "airflow"
]


@st.cache_resource(show_spinner=False)
def load_ml_assets():
    """
    Load pre-trained Regression Model and TF-IDF Vectorizer.
    Returns (model, vectorizer, is_loaded).
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            return model, vectorizer, True
        except Exception as e:
            st.warning(f"Error loading model files: {e}")
            return None, None, False
    return None, None, False


# -----------------------------------------------------------------------------
# 3. TEXT PROCESSING & FEATURE EXTRACTION
# -----------------------------------------------------------------------------
def extract_pdf_text(uploaded_file) -> str:
    """Extract raw text from PDF files using PyPDF2."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""


def extract_docx_text(uploaded_file) -> str:
    """Extract raw text from DOCX files using python-docx."""
    try:
        document = Document(uploaded_file)
        text = ""
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""


def clean_text(text: str) -> str:
    """
    Clean and preprocess input text:
    - Lowercase conversion
    - Retain relevant alphanumeric and technical characters (+, #, ., -)
    - Normalize whitespaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9+#.\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_skills(text: str) -> list:
    """
    Extract technical skills from text using boundary-safe regex matching.
    """
    text_clean = " " + text.lower() + " "
    found = []
    # Sort skills by length descending to match composite terms first
    sorted_skills = sorted(SKILLS_DATABASE, key=len, reverse=True)
    
    for skill in sorted_skills:
        escaped = re.escape(skill)
        if skill == "c":
            pattern = r'(?<![a-zA-Z0-9+#])c(?![a-zA-Z0-9+#])'
        elif skill == "r":
            pattern = r'(?<![a-zA-Z0-9])r(?![a-zA-Z0-9])'
        else:
            pattern = r'(?<![a-zA-Z0-9])' + escaped + r'(?![a-zA-Z0-9])'
            
        if re.search(pattern, text_clean):
            found.append(skill)
            
    return sorted(list(set(found)))


def compute_features(resume_clean: str, jd_clean: str, vectorizer: TfidfVectorizer) -> tuple:
    """
    Compute Cosine Similarity and regression feature vector:
    Returns (cosine_score_pct, feature_vector, matched_skills, missing_skills)
    """
    # 1. Vectorize text
    vec_res = vectorizer.transform([resume_clean])
    vec_jd = vectorizer.transform([jd_clean])
    
    # 2. Cosine Similarity
    cos_sim = float(cosine_similarity(vec_res, vec_jd)[0][0])
    cos_score_pct = round(cos_sim * 100, 2)
    
    # 3. Jaccard & Token Overlap
    res_tokens = set(resume_clean.split())
    jd_tokens = set(jd_clean.split())
    intersection = len(res_tokens & jd_tokens)
    union = len(res_tokens | jd_tokens) if len(res_tokens | jd_tokens) > 0 else 1
    jaccard = intersection / union
    
    # 4. JD Token Coverage
    jd_cov = intersection / len(jd_tokens) if len(jd_tokens) > 0 else 0.0
    
    # 5. Skill Extraction
    res_skills = extract_skills(resume_clean)
    jd_skills = extract_skills(jd_clean)
    matched_skills = sorted(list(set(res_skills) & set(jd_skills)))
    missing_skills = sorted(list(set(jd_skills) - set(res_skills)))
    skill_ratio = len(matched_skills) / len(jd_skills) if len(jd_skills) > 0 else 0.0
    
    # 6. Length Ratio
    len_ratio = min(len(resume_clean), len(jd_clean)) / max(len(resume_clean), len(jd_clean), 1)
    
    feature_vector = np.array([[cos_sim, jaccard, jd_cov, skill_ratio, len(matched_skills), len_ratio]])
    return cos_score_pct, feature_vector, matched_skills, missing_skills


def fallback_calculate_match(resume_clean: str, jd_clean: str) -> float:
    """Fallback Cosine Similarity calculation if ML model artifacts are missing."""
    docs = [resume_clean, jd_clean]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(docs)
    sim = cosine_similarity(matrix[0:1], matrix[1:2])
    return round(float(sim[0][0]) * 100, 2)


# -----------------------------------------------------------------------------
# 4. SIDEBAR - MODEL STATUS & PRESETS
# -----------------------------------------------------------------------------
model, vectorizer, is_model_loaded = load_ml_assets()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=64)
    st.markdown("### ⚙️ System Status")
    
    if is_model_loaded:
        st.success("🟢 **ML Model Active**\nRandomForestRegressor + TF-IDF")
    else:
        st.warning("🟡 **Fallback Active**\nRun `train_model.py` to activate ML Regression.")
        
    st.markdown("---")
    st.markdown("### 🧪 Quick Demo Presets")
    st.caption("Load sample profiles to quickly test the matching engine:")

    sample_choice = st.selectbox(
        "Select Demo Scenario:",
        [
            "None (Upload Custom)",
            "High Match: Senior Python Developer",
            "High Match: Data Scientist",
            "Medium Match: Frontend Developer applying for Full Stack",
            "Low Match: Graphic Designer applying for AI Engineer"
        ]
    )

    sample_resumes = {
        "High Match: Senior Python Developer": (
            "Senior Python Developer with 5 years experience in building REST APIs using Python, Django, FastAPI, Flask, PostgreSQL, Docker, Redis, Celery, and Git. Proficient in Pytest, CI/CD, microservices architecture, and AWS EC2.",
            "We are seeking an experienced Python Developer to build scalable backend services and RESTful APIs. Must have strong skills in Python, Django or FastAPI, PostgreSQL, Redis, Celery, Docker, unit testing, and Git version control."
        ),
        "High Match: Data Scientist": (
            "Data Scientist with 4 years experience in Python, Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, SQL, statistical modeling, Exploratory Data Analysis, and Tableau. Built predictive models and customer segmentation algorithms.",
            "Seeking a Data Scientist proficient in Python, SQL, statistical modeling, machine learning with Scikit-learn, PyTorch or TensorFlow, Pandas, and data visualization (Tableau/Power BI)."
        ),
        "Medium Match: Frontend Developer applying for Full Stack": (
            "Frontend Web Developer with 3 years of experience in React, JavaScript, HTML5, CSS3, Tailwind CSS, Redux, and basic Node.js backend. Built single page applications and responsive UI.",
            "Looking for a Full Stack Software Engineer to build complex backend microservices in Python, Django, PostgreSQL, Docker, Redis caching, CI/CD pipelines, and React frontend."
        ),
        "Low Match: Graphic Designer applying for AI Engineer": (
            "Graphic Designer with expert proficiency in Adobe Photoshop, Illustrator, Figma, brand identity design, wireframing, typography, and creative media editing.",
            "We are seeking an AI Engineer with expertise in Generative AI, LLMs, LangChain, RAG architecture, vector search, Python, PyTorch, and model deployment in Docker."
        )
    }

    st.markdown("---")
    st.markdown("### 📚 Project Metadata")
    st.markdown("""
    - **Architecture**: NLP + ML Regression
    - **Vectorization**: TF-IDF (1-2 ngrams)
    - **Similarity**: Cosine Similarity
    - **Regression**: Random Forest
    - **Target Metric**: Match Score (0-100%)
    """)

# -----------------------------------------------------------------------------
# 5. MAIN HEADER & USER INTERFACE
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🎯 AI-Powered Resume vs Job Description Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze resume alignment, predict match percentage with Machine Learning, and uncover skill gaps with tailored recommendations.</div>', unsafe_allow_html=True)

# Main Inputs layout
col_input1, col_input2 = st.columns([1, 1], gap="large")

# Pre-populate if demo preset is selected
preset_resume = ""
preset_jd = ""
if sample_choice != "None (Upload Custom)" and sample_choice in sample_resumes:
    preset_resume, preset_jd = sample_resumes[sample_choice]

with col_input1:
    st.markdown("#### 📄 1. Upload or Provide Resume")
    uploaded_resume = st.file_uploader(
        "Upload Resume (PDF or DOCX format)",
        type=["pdf", "docx"],
        help="Upload candidate resume in PDF or DOCX format."
    )
    
    manual_resume_text = ""
    if preset_resume and uploaded_resume is None:
        manual_resume_text = st.text_area(
            "📋 Loaded Demo Resume Text:",
            value=preset_resume,
            height=180
        )

with col_input2:
    st.markdown("#### 💼 2. Job Description")
    job_description = st.text_area(
        "Paste the Target Job Description here:",
        value=preset_jd if preset_jd else "",
        height=220,
        placeholder="Paste full job description including requirements, technical stack, and responsibilities..."
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. ANALYSIS & PREDICTION PIPELINE
# -----------------------------------------------------------------------------
analyze_btn = st.button("🚀 Analyze & Match Resume", type="primary", use_container_width=True)

if analyze_btn:
    # 1. Validation
    raw_resume_text = ""
    
    if uploaded_resume is not None:
        file_name = uploaded_resume.name.lower()
        with st.spinner("Extracting text from resume file..."):
            if file_name.endswith(".pdf"):
                raw_resume_text = extract_pdf_text(uploaded_resume)
            elif file_name.endswith(".docx"):
                raw_resume_text = extract_docx_text(uploaded_resume)
            else:
                st.error("Unsupported file format. Please upload a PDF or DOCX file.")
    elif manual_resume_text.strip():
        raw_resume_text = manual_resume_text
        
    if not raw_resume_text.strip():
        st.warning("⚠️ Please upload a resume file (PDF/DOCX) or select a demo scenario.")
    elif not job_description.strip():
        st.warning("⚠️ Please provide a job description to analyze against.")
    else:
        with st.spinner("🧠 Running NLP preprocessing, TF-IDF vectorization, and ML regression model..."):
            # 2. Text Preprocessing & Cleaning
            resume_clean = clean_text(raw_resume_text)
            jd_clean = clean_text(job_description)
            
            # 3. Model Prediction or Fallback
            if is_model_loaded and vectorizer is not None and model is not None:
                cos_score, feat_vec, matched_skills, missing_skills = compute_features(
                    resume_clean, jd_clean, vectorizer
                )
                pred_raw = model.predict(feat_vec)[0]
                ml_score = round(float(np.clip(pred_raw, 0, 100)), 2)
                # Final calibrated match score
                final_score = ml_score
            else:
                cos_score = fallback_calculate_match(resume_clean, jd_clean)
                ml_score = cos_score
                final_score = cos_score
                res_skills = extract_skills(resume_clean)
                jd_skills = extract_skills(jd_clean)
                matched_skills = sorted(list(set(res_skills) & set(jd_skills)))
                missing_skills = sorted(list(set(jd_skills) - set(res_skills)))

            # 4. Determine Match Tier & Styling
            if final_score >= 80:
                tier_label = "Excellent Match"
                tier_color = "#10B981"
                tier_icon = "🌟"
                tier_alert = st.success
                tier_desc = "Outstanding alignment! The candidate profile strongly matches the technical stack and core competencies requested."
            elif final_score >= 60:
                tier_label = "Good Match"
                tier_color = "#3B82F6"
                tier_icon = "👍"
                tier_alert = st.info
                tier_desc = "Strong candidate profile with solid foundational skills. A few complementary skills or certifications could make this application top-tier."
            elif final_score >= 40:
                tier_label = "Moderate Match"
                tier_color = "#F59E0B"
                tier_icon = "⚠️"
                tier_alert = st.warning
                tier_desc = "Moderate fit. The resume shares foundational topics but is missing key technical tools and role-specific requirements."
            else:
                tier_label = "Low Match"
                tier_color = "#EF4444"
                tier_icon = "❌"
                tier_alert = st.error
                tier_desc = "Low alignment. Significant skill and domain gaps identified between the candidate's resume and the job prerequisites."

        # ---------------------------------------------------------------------
        # 7. DASHBOARD DISPLAY
        # ---------------------------------------------------------------------
        st.markdown("### 📊 Match Analysis Results")
        
        # Top 3 Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📊 Final Match Score</div>
                <div class="metric-val" style="color: {tier_color};">{final_score}%</div>
                <div style="font-size: 0.9rem; font-weight: 600; color: {tier_color};">{tier_icon} {tier_label}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🧠 Cosine Similarity Score</div>
                <div class="metric-val" style="color: #6366F1;">{cos_score}%</div>
                <div style="font-size: 0.85rem; color: #94A3B8;">Vector Space Angle Metric</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📈 ML Regression Predicted Score</div>
                <div class="metric-val" style="color: #EC4899;">{ml_score}%</div>
                <div style="font-size: 0.85rem; color: #94A3B8;">Random Forest Evaluator</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Progress Bar & Tier interpretation
        st.progress(int(final_score))
        
        # Tier Alert Box
        tier_alert(f"**{tier_icon} {tier_label} ({final_score}%)**: {tier_desc}")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 8. SKILL ANALYSIS SECTION
        # ---------------------------------------------------------------------
        st.markdown("### 🛠️ Skills & Competency Breakdown")
        
        col_skills1, col_skills2 = st.columns(2, gap="medium")
        
        with col_skills1:
            st.markdown(f"#### ✅ Matched Skills ({len(matched_skills)})")
            if matched_skills:
                badges_html = "".join([f'<span class="badge-chip badge-matched">✓ {skill.title()}</span>' for skill in matched_skills])
                st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)
            else:
                st.info("No direct technical skill matches found in our database taxonomy.")

        with col_skills2:
            st.markdown(f"#### ❌ Missing Skills ({len(missing_skills)})")
            if missing_skills:
                badges_html = "".join([f'<span class="badge-chip badge-missing">✗ {skill.title()}</span>' for skill in missing_skills])
                st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)
            else:
                st.success("🎉 All target skills from the job description appear to be covered!")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 9. PERSONALIZED AI RECOMMENDATIONS
        # ---------------------------------------------------------------------
        st.markdown("### 💡 Personalized AI Recommendations")
        
        rec_col1, rec_col2 = st.columns([1.5, 1], gap="medium")
        
        with rec_col1:
            st.markdown("##### 📌 Strategic Action Items")
            if missing_skills:
                top_missing = ", ".join([f"**{s.title()}**" for s in missing_skills[:5]])
                st.markdown(f"- **Bridge Critical Skill Gaps**: Prioritize adding projects or experience demonstrating knowledge of {top_missing}.")
            
            if final_score >= 80:
                st.markdown("- **Optimize Impact Metrics**: Quantify your achievements (e.g., *'improved pipeline latency by 35%'*, *'scaled service to 10k RPS'*) to make your strong resume stand out even more.")
                st.markdown("- **Prepare for Domain Deep-Dives**: Review system design and architectural trade-offs related to the matched technologies.")
            elif final_score >= 60:
                st.markdown("- **Align Resume Terminology**: Ensure technical terms in your resume mirror the exact keywords used in the job description.")
                st.markdown("- **Highlight Relevant Projects**: Add a dedicated *Projects* or *Technical Highlights* section spotlighting the required stack.")
            elif final_score >= 40:
                st.markdown("- **Tailor Your Summary & Skills**: Re-organize your resume to put matching technical skills and relevant tools front-and-center.")
                st.markdown("- **Upskill in Core Areas**: Take focused courses or build hands-on repositories showcasing the missing tools.")
            else:
                st.markdown("- **Fundamental Re-alignment**: The target position requires a significantly different core skill set. Focus on building foundational experience in this domain.")
                st.markdown("- **Build Capstone Projects**: Develop portfolio projects demonstrating end-to-end implementation of the required tools.")

        with rec_col2:
            st.markdown("##### 📈 Match Score Breakdown")
            st.markdown(f"""
            - **Skill Coverage**: {len(matched_skills)} / {max(len(matched_skills) + len(missing_skills), 1)} skills matched
            - **NLP Cosine Angle**: `{cos_score}%`
            - **ML Non-Linear Score**: `{ml_score}%`
            - **Target Range**: `80-100%` for top priority interviews
            """)

# -----------------------------------------------------------------------------
# 10. EDUCATIONAL SECTION - HOW THE AI MODEL WORKS
# -----------------------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
with st.expander("ℹ️ How the AI & Machine Learning Model Works", expanded=False):
    st.markdown("""
    ### 🔬 Architectural Workflow & Methodology
    
    This system predicts candidate-job fit using a hybrid approach combining Natural Language Processing (NLP), Vector Space Similarity, and Supervised Machine Learning:
    """)
    
    st.markdown("""
    <div class="workflow-step">
        <b>1. Text Extraction & Ingestion</b><br>
        Extracts unstructured raw text from uploaded PDF documents (via <code>PyPDF2</code>) or DOCX files (via <code>python-docx</code>).
    </div>
    <div class="workflow-step">
        <b>2. NLP Preprocessing & Normalization</b><br>
        Converts text to lowercase, handles technical character preservation (e.g. <i>C++, C#, Node.js, .NET</i>), strips non-informative punctuation, and normalizes whitespaces.
    </div>
    <div class="workflow-step">
        <b>3. TF-IDF Vectorization (Term Frequency - Inverse Document Frequency)</b><br>
        Transforms text into numerical feature vectors capturing unigrams and bigrams weighted by document frequency:
        $$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\log\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$$
    </div>
    <div class="workflow-step">
        <b>4. Cosine Similarity in High-Dimensional Space</b><br>
        Computes the cosine of the angle between the resume vector ($\\mathbf{u}$) and job description vector ($\\mathbf{v}$):
        $$\\text{Cosine Similarity}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\| \\|\\mathbf{v}\\|}$$
    </div>
    <div class="workflow-step">
        <b>5. Random Forest Regression Model</b><br>
        Feeds Cosine Similarity along with Jaccard token overlap, keyword coverage, and skill alignment ratios into a trained <code>RandomForestRegressor</code> to predict a robust non-linear match score calibrated between <b>0% and 100%</b>.
    </div>
    <div class="workflow-step">
        <b>6. Boundary-Aware Skill Extraction & Diagnostics</b><br>
        Extracts matched and missing technical skills against an extensive taxonomy to give actionable, transparent feedback.
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Educational Note**: The machine learning model was trained on a synthetic paired educational dataset spanning 7+ engineering categories (Data Science, ML Engineering, Python Development, Full-Stack, Web Development, Data Analysis, and AI Engineering).")

# Footer
st.markdown("<br><div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>AI Resume Matcher • Built with Streamlit, Scikit-Learn & Python</div>", unsafe_allow_html=True)