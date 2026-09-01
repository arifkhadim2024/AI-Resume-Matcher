"""
train_model.py - Machine Learning Model Training Pipeline for AI Resume Matcher

This script trains a regression model to predict the match score (0-100%) between
a candidate's resume and a job description using TF-IDF Vectorization,
Cosine Similarity, and engineered NLP features.

Workflow:
1. Load dataset (data/resume_jd_dataset.csv)
2. Preprocess & clean text
3. Fit TF-IDF Vectorizer on corpus
4. Compute Cosine Similarity and composite match features
5. Train a Regression Model (RandomForestRegressor)
6. Evaluate model performance (MAE, RMSE, R² Score)
7. Save model and vectorizer artifacts to model/ directory
"""

import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "resume_jd_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "regression_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# Expanded technical skills vocabulary for skill extraction & overlap feature
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


def clean_text(text: str) -> str:
    """
    Clean and preprocess raw text:
    - Converts to lowercase
    - Preserves relevant technical characters (+, #, ., -)
    - Replaces special characters with space
    - Normalizes extra whitespaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9+#.\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_skills(text: str) -> set:
    """
    Extract technical skills from text with word-boundary awareness.
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
            
    return set(found)


def extract_features(resume_clean: str, jd_clean: str, vectorizer: TfidfVectorizer) -> list:
    """
    Compute input features for the regression model:
    1. TF-IDF Cosine Similarity (Anchor feature: 0 to 1)
    2. Jaccard Word Similarity (Token overlap ratio)
    3. Job Description Keyword Coverage
    4. Skill Match Ratio (Matched skills / Total JD skills)
    5. Count of Matched Skills
    6. Character Length Ratio
    """
    vec_res = vectorizer.transform([resume_clean])
    vec_jd = vectorizer.transform([jd_clean])
    
    # 1. Cosine similarity
    cos_sim = float(cosine_similarity(vec_res, vec_jd)[0][0])
    
    # 2. Token sets & Jaccard
    res_tokens = set(resume_clean.split())
    jd_tokens = set(jd_clean.split())
    intersection = len(res_tokens & jd_tokens)
    union = len(res_tokens | jd_tokens) if len(res_tokens | jd_tokens) > 0 else 1
    jaccard = intersection / union
    
    # 3. JD Coverage
    jd_cov = intersection / len(jd_tokens) if len(jd_tokens) > 0 else 0.0
    
    # 4. & 5. Skill overlap
    res_skills = extract_skills(resume_clean)
    jd_skills = extract_skills(jd_clean)
    matched_skills = res_skills & jd_skills
    skill_ratio = len(matched_skills) / len(jd_skills) if len(jd_skills) > 0 else 0.0
    
    # 6. Length ratio
    len_ratio = min(len(resume_clean), len(jd_clean)) / max(len(resume_clean), len(jd_clean), 1)
    
    return [cos_sim, jaccard, jd_cov, skill_ratio, len(matched_skills), len_ratio]


def train_pipeline():
    """
    Main training routine:
    Loads data, trains vectorizer & regression model, evaluates, and saves artifacts.
    """
    print("=" * 60)
    print("🚀 AI RESUME MATCHER - ML MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load Dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please generate dataset first.")
        
    print(f"[*] Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[+] Loaded {len(df)} resume-job description pairs.")
    print(f"[+] Columns: {list(df.columns)}")
    
    # 2. Text Preprocessing
    print("[*] Preprocessing and cleaning text...")
    df['resume_clean'] = df['resume_text'].apply(clean_text)
    df['jd_clean'] = df['job_description'].apply(clean_text)
    
    # 3. Fit TF-IDF Vectorizer
    print("[*] Fitting TF-IDF Vectorizer (ngram_range=(1,2), max_features=5000)...")
    corpus = pd.concat([df['resume_clean'], df['jd_clean']]).unique()
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    vectorizer.fit(corpus)
    print(f"[+] TF-IDF vocabulary size: {len(vectorizer.vocabulary_)} terms.")
    
    # 4. Extract Features
    print("[*] Computing Cosine Similarity and composite NLP features...")
    X_list = []
    for _, row in df.iterrows():
        feats = extract_features(row['resume_clean'], row['jd_clean'], vectorizer)
        X_list.append(feats)
        
    X = np.array(X_list)
    y = df['match_score'].values.astype(float)
    
    # 5. Split Dataset
    print("[*] Splitting dataset into 80% Train and 20% Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[+] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    
    # 6. Train Regression Model
    print("[*] Training RandomForestRegressor (n_estimators=150, max_depth=8)...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 7. Evaluate Model Performance
    print("\n" + "=" * 60)
    print("📊 MODEL EVALUATION METRICS")
    print("=" * 60)
    
    y_pred_train = np.clip(model.predict(X_train), 0, 100)
    y_pred_test = np.clip(model.predict(X_test), 0, 100)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"{'Metric':<25} | {'Training Set':<15} | {'Testing Set':<15}")
    print("-" * 60)
    print(f"{'Mean Absolute Error (MAE)':<25} | {mae_train:<15.2f} | {mae_test:<15.2f}")
    print(f"{'Root Mean Squared (RMSE)':<25} | {rmse_train:<15.2f} | {rmse_test:<15.2f}")
    print(f"{'R² Score (Variance)':<25} | {r2_train:<15.4f} | {r2_test:<15.4f}")
    print("=" * 60)
    
    # 8. Save Artifacts
    print("\n[*] Saving model artifacts...")
    joblib.dump(model, MODEL_PATH)
    print(f"[+] Saved Regression Model to: {MODEL_PATH}")
    
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[+] Saved TF-IDF Vectorizer to: {VECTORIZER_PATH}")
    
    print("\n✅ Model training and export completed successfully!")
    return model, vectorizer


if __name__ == "__main__":
    train_pipeline()
