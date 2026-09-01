# 🎯 AI-Powered Resume vs Job Description Matcher

An end-to-end Machine Learning and Natural Language Processing (NLP) system that evaluates how well a candidate's resume matches a target Job Description. The application parses resumes in **PDF** and **DOCX** formats, preprocesses and vectorizes text using **TF-IDF**, calculates **Cosine Similarity**, predicts a calibrated match score (0–100%) via a trained **Random Forest Regression model**, extracts technical skills, and provides personalized recommendations.

---

## 📌 Project Overview

- **Project Title**: Resume vs Job Description Matching Score
- **Primary Objective**: Build an AI/ML system that analyzes candidate resumes against job descriptions and provides an accurate, explainable match score with skill gap analysis and improvement suggestions.
- **Tech Stack**: Python 3.x, Streamlit, Scikit-Learn, Pandas, NumPy, PyPDF2, python-docx, Joblib.

---

## 🏗️ System Architecture & Workflow

```
Candidate Resume (PDF / DOCX)               Target Job Description
            │                                         │
            ▼                                         ▼
   Extract Text (PyPDF2 / docx)             Raw Text Input Area
            │                                         │
            └────────────────────┬────────────────────┘
                                 ▼
                     Text Cleaning & Preprocessing
              (Lowercasing, Token Preservation, Normalization)
                                 │
                                 ▼
                      TF-IDF Vectorization
                   (Unigrams + Bigrams, Sublinear TF)
                                 │
                                 ▼
                      Cosine Similarity Angle
                                 │
                                 ▼
                    Multi-Feature Extraction
       (Cosine Sim, Jaccard Index, Skill Overlap, Length Ratio)
                                 │
                                 ▼
                 Trained Regression Model (Random Forest)
                                 │
                                 ▼
                    Final Match Score (0 - 100%)
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
   Skill Gap Diagnostics                     Personalized AI
(Matched vs Missing Skills)                  Recommendations
```

---

## 📂 Project Directory Structure

```
AI_Resume_Matcher/
│
├── AI_Resume_Graph_Prediction.ipynb # Jupyter Notebook (Graph Analytics & ML Prediction)
├── app.py                           # Streamlit Web Application (UI & Inference)
├── train_model.py                   # ML Model Training & Evaluation Pipeline
├── requirements.txt                 # Project Dependencies
├── README.md                        # Comprehensive Documentation
├── .gitignore                       # Git ignore rules for virtualenvs & caches
│
├── data/
│   └── resume_jd_dataset.csv        # Paired Resume-JD Educational Dataset
│
└── model/
    ├── regression_model.pkl         # Trained Random Forest Regression Model
    └── tfidf_vectorizer.pkl         # Fitted TF-IDF Vectorizer
```

---

## 📊 Dataset Specifications

The dataset is stored in `data/resume_jd_dataset.csv` and contains paired resume-job description examples designed for machine learning model training:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `resume_text` | String | Extracted / text content of the candidate's resume. |
| `job_description` | String | Target job description containing requirements and stack. |
| `match_score` | Float / Int | Ground truth compatibility score ranging from `0` to `100`. |

### Job Roles Covered:
1. **Data Scientist**
2. **Machine Learning Engineer**
3. **Python Developer**
4. **Software Engineer / Full Stack**
5. **Web Developer (Frontend / Backend)**
6. **Data Analyst**
7. **AI / GenAI Engineer**

### Score Distribution:
- **High Match (80 - 100%)**: Strong alignment in core technologies, domain experience, and tools.
- **Medium Match (50 - 79%)**: Partial overlap, adjacent tech stacks, or transferable technical skills.
- **Low Match (0 - 49%)**: Mismatched technical domains or missing prerequisite tools.

> **Note**: This dataset is a realistic synthetic educational dataset generated for academic and demonstration purposes.

---

## 🧠 Machine Learning & NLP Methodology

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)
Text documents are mapped into high-dimensional vector spaces using TF-IDF representation:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### 2. Cosine Similarity
Measures the directional alignment between the resume vector $\mathbf{u}$ and job description vector $\mathbf{v}$:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

### 3. Feature Engineering & Regression
The regression model ingests composite features:
- **TF-IDF Cosine Similarity** (Anchor feature: 0 to 1)
- **Jaccard Token Overlap Ratio** ($\frac{|A \cap B|}{|A \cup B|}$)
- **Job Description Keyword Coverage**
- **Skill Match Ratio** ($\frac{|\text{Matched Skills}|}{|\text{JD Skills}|}$)
- **Matched Skill Count**
- **Document Length Ratio**

A `RandomForestRegressor` is trained to capture non-linear interactions and output an interpretable score clipped between **0% and 100%**.

---

## 📈 Model Evaluation Metrics

Evaluated on an 80/20 train-test split:

| Evaluation Metric | Training Set | Testing Set |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | `2.94` | `8.39` |
| **Root Mean Squared Error (RMSE)** | `3.94` | `13.27` |
| **Coefficient of Determination ($R^2$)** | `0.9851` | `0.8172` |

---

## 🚀 Quickstart & Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/arifkhadim2024/AI-Resume-Matcher.git
cd AI_Resume_Matcher
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Machine Learning Model
Generate the model artifacts (`regression_model.pkl` and `tfidf_vectorizer.pkl`):
```bash
python train_model.py
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

### Step 6: Run the Graph Prediction & Analytics Notebook (Optional)
Launch Jupyter to explore interactive model training, residual plots, and bipartite knowledge graphs:
```bash
jupyter notebook AI_Resume_Graph_Prediction.ipynb
```

---

## 🖥️ Streamlit Application Features

- **Document Ingestion**: Upload resumes in `.pdf` or `.docx` format with instant text extraction.
- **Interactive Preset Loader**: Quick-test preloaded profiles (Data Scientist, Senior Python Dev, Frontend Dev, Mismatch).
- **Match Score Dashboard**:
  - 📊 **Final Match Score**: Calibrated 0–100% score.
  - 🧠 **Cosine Similarity Score**: High-dimensional vector angle.
  - 📈 **ML Regression Score**: Random Forest prediction.
- **Score Tier System**:
  - `80 - 100%`: 🌟 **Excellent Match**
  - `60 - 79%`: 👍 **Good Match**
  - `40 - 59%`: ⚠️ **Moderate Match**
  - `0 - 39%`: ❌ **Low Match**
- **Skill Gap Analysis**: Visual badge chips displaying **Matched Skills** vs. **Missing Skills**.
- **Personalized Recommendations**: Dynamic action items tailored to the score tier and specific missing keywords.
- **Educational Explainer**: Interactive breakdown of how TF-IDF, Cosine Similarity, and Random Forest Regression function.

---

## 📓 Jupyter Notebook & Graph Prediction Features

The notebook [`AI_Resume_Graph_Prediction.ipynb`](AI_Resume_Graph_Prediction.ipynb) includes:
1. **Interactive Exploratory Data Analysis**: Word count distributions, score histograms, and correlation heatmaps.
2. **Comparative Model Training**: Benchmarks `RandomForestRegressor`, `GradientBoostingRegressor`, and `RidgeRegression`.
3. **Predictive Diagnostic Graphs**:
   - **Actual vs. Predicted Plot**: Evaluates linearity and test $R^2$ score against the ideal 45° line.
   - **Residual Analysis Plot**: Residual error dispersion across prediction ranges.
   - **Feature Importance Chart**: Gini importance breakdown for Cosine Similarity, Skill Ratio, Jaccard Index, and Length Ratio.
   - **Model Comparison Metrics**: Side-by-side MAE and $R^2$ charts.
4. **Bipartite Skill & Knowledge Graph Analytics**:
   - Models candidate skills and job requirements as a bipartite graph $G = (V_{\text{candidate}}, V_{\text{job}}, E)$ using `NetworkX`.
   - Visual color-coding: 🟢 Matched Skills, 🔴 Missing Prerequisites, 🔵 Additional Candidate Strengths.
   - Calculates graph density, node degree, and requirement coverage ratios.
5. **Dual-Graph Inference Visualizer**:
   - Live function `predict_and_plot_resume_match(resume, jd)` generating both a **Bipartite Skill Graph** and a **Multivariate Competency Radar Chart**.
   - Pre-evaluated across 3 hiring scenarios (Senior Python Dev, Frontend Dev, Designer to AI Engineer).

---

## 🛠️ Git Version Control Commands

To stage, commit, and push your upgraded project to GitHub:

```bash
# 1. Stage all changes
git add .

# 2. Commit the changes
git commit -m "Upgrade AI Resume Matcher with ML regression"

# 3. Push to your repository
git push
```

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
