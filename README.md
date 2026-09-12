# 🤖 PyBot — College Subjects & AI Tutor Chatbot (with RAG)

**Graduation Project — Data Science & AI Diploma**
Category: *Chatbot for college subjects*

PyBot combines three layers of AI:
1. A **trained ML classifier** (TF-IDF + Logistic Regression) that routes each message
2. Instant answers for administrative FAQs (exam dates, deadlines, office hours...) straight from the model — no API call needed
3. A **RAG (Retrieval-Augmented Generation) pipeline** for academic questions: relevant course-note chunks are retrieved from a knowledge base and used to ground the LLM's (Groq) answer, instead of relying only on the LLM's general knowledge

---

## 👥 Authors
- **Mazen Zayan**
- **Karam Zayan**

---

## 🎯 Project Summary

| | |
|---|---|
| **Task type** | Text classification (NLP) + LLM-based tutoring |
| **Dataset** | Self-built, 221 labeled student messages across 16 intents |
| **Model** | TF-IDF + Logistic Regression (tuned via GridSearchCV) |
| **Baseline** | Multinomial Naive Bayes |
| **Deployment** | Flask web app, model runs locally, LLM calls via Groq API |

---

## 🗂️ Dataset

`data/college_chatbot_dataset.csv` — 221 example student messages labeled with one of 16 intents:

**Administrative / FAQ intents** (answered instantly by the trained model):
`greeting`, `goodbye`, `thanks`, `exam_schedule`, `assignment_deadline`, `office_hours`, `course_registration`, `grading_system`, `library_hours`, `technical_support`

**Academic / tutoring intents** (forwarded to the LLM for a full explanation):
`python_basics`, `data_structures`, `databases`, `machine_learning`, `statistics`, `general_help`

The dataset is intentionally **imbalanced** (8–22 examples per intent) to give the preprocessing pipeline a genuine imbalance problem to detect and correct.

---

## 🔬 Preprocessing Pipeline (`preprocessing_and_training.py`)

All 8 required steps are implemented, in order:

1. **Load and Explore the Dataset** — shape, dtypes, class list, sample rows
2. **Check for and Handle Duplicates** — exact-text duplicates detected and dropped
3. **Handle Missing Values** — checked per column, none found
4. **Check Outliers and Handle Them** — message length (word count) outliers via IQR, visualized with a boxplot
5. **Visualizations** — class distribution, word-count histogram, top-word frequency chart, data-quality summary (saved to `outputs/plots/`)
6. **Check Imbalance and Handle It** — minority/majority ratio computed (0.36 → imbalanced); balanced via random oversampling to equal class sizes
7. **Address Overfitting and Underfitting** — baseline vs. tuned model compared, 5-fold cross-validation, learning curve plotted
8. **Evaluate Model Performance** — accuracy, precision, recall, F1 (macro), confusion matrix, full classification report

Full results are written to `outputs/REPORT.md` every time the script runs, and every chart is saved to `outputs/plots/`.

### Headline results
- **Baseline (Naive Bayes)** test accuracy: **0.79**
- **Final model (Logistic Regression, tuned)** test accuracy: **0.79**, macro F1: **0.81**
- **5-fold CV accuracy:** 0.87 ± 0.04

*(Re-run the script any time to regenerate the dataset split, plots, and report — see below.)*

---

## 🧠 How the Chatbot Uses the Model + RAG

```
User message
     │
     ▼
TF-IDF vectorizer → Logistic Regression classifier
     │
     ├─ Confidence ≥ 0.35 AND intent is an FAQ intent
     │        → answer instantly from data/faq_responses.csv (no API call)
     │
     └─ Otherwise (academic question / low confidence)
              │
              ▼
        RAG retriever searches data/knowledge_base.json
        (31 course-note chunks: Python, Data Structures,
         Databases, Machine Learning, Statistics)
              │
              ▼
        Top-3 relevant chunks injected into the system prompt
              │
              ▼
        Groq LLM generates the final answer, grounded in
        the retrieved course notes
              │
              ▼
        📚 Explanation / 💻 Example / 📝 Practice / ✅ Tip
```

This means the app is never "just an API wrapper" — a trained classifier decides how to route every message, and a retrieval step grounds academic answers in real course content instead of the LLM's general knowledge alone.

### RAG knowledge base
- `data/build_knowledge_base.py` — generates `data/knowledge_base.json` (31 chunks across 5 subjects). Replace/extend the text in this file with your actual lecture notes or slides content whenever you have them — no other code needs to change.
- `build_retriever_index.py` — indexes the knowledge base with TF-IDF (works immediately, no GPU/internet required).
- `retriever.py` — used by `app.py` at request time to fetch the most relevant chunks for a question.

### Optional upgrade: real semantic embeddings (Google Colab)
The default retriever uses TF-IDF (keyword-based) similarity, which works well for a course-notes knowledge base but can miss paraphrased questions. To upgrade to dense semantic embeddings:
1. Open `colab_generate_embeddings.ipynb` in Google Colab.
2. Upload `data/knowledge_base.json` when prompted.
3. Run all cells — it uses `sentence-transformers` (`all-MiniLM-L6-v2`) to embed every chunk.
4. Download the resulting `kb_embeddings.npy` and place it in `model/`.
5. Restart `app.py` — `retriever.py` automatically detects and prefers the dense embeddings over TF-IDF.

No code changes needed either way — this is a drop-in upgrade.

---

## 🚀 Getting Started

### 1. Create and activate a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your `.env` file
Copy `.env.example` to `.env` and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
FLASK_DEBUG=false
```
Get a key from [console.groq.com/keys](https://console.groq.com/keys).

### 4. Generate the dataset and knowledge base (already included, but reproducible)
```bash
python data/generate_dataset.py
python data/build_knowledge_base.py
```

### 5. Run the preprocessing + training pipeline
```bash
python preprocessing_and_training.py
python build_retriever_index.py
```
This creates:
- `model/intent_classifier.pkl` and `model/tfidf_vectorizer.pkl` (intent classifier)
- `model/kb_tfidf_vectorizer.pkl`, `model/kb_tfidf_matrix.pkl`, `model/kb_chunks.pkl` (RAG retriever)
- `outputs/plots/*.png` (all 8 EDA/evaluation charts)
- `outputs/REPORT.md` (full written report of every step)

### 6. Run the app
```bash
python app.py
```
Visit **http://127.0.0.1:5000**.

---

## 📁 Project Structure

```
college_chatbot/
├── app.py                          # Flask app (intent model + RAG + LLM)
├── retriever.py                     # RAG retrieval (TF-IDF or dense embeddings)
├── preprocessing_and_training.py    # Full 8-step pipeline + intent model training
├── build_retriever_index.py         # Indexes the knowledge base for RAG
├── colab_generate_embeddings.ipynb  # Optional: real embeddings via Google Colab
├── requirements.txt
├── .env.example
├── data/
│   ├── generate_dataset.py          # Builds the labeled intent dataset
│   ├── college_chatbot_dataset.csv  # 221 labeled examples, 16 intents
│   ├── faq_responses.csv            # Canned answers for FAQ intents
│   ├── build_knowledge_base.py      # Builds the RAG knowledge base
│   └── knowledge_base.json          # 31 course-note chunks, 5 subjects
├── model/
│   ├── intent_classifier.pkl        # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer (intents)
│   ├── kb_tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer (RAG)
│   ├── kb_tfidf_matrix.pkl          # RAG document-term matrix
│   ├── kb_chunks.pkl                # RAG chunks (cached from knowledge_base.json)
│   └── kb_embeddings.npy            # Optional — dense embeddings from Colab
├── outputs/
│   ├── REPORT.md                    # Full pipeline report (auto-generated)
│   └── plots/                       # All EDA + evaluation charts
└── templates/
    └── index.html                   # Chat UI
```

---

## 🔒 Security Notes
- The Groq API key lives only in `.env`, never hardcoded in `app.py`.
- `.env` is excluded from version control via `.gitignore`.
- If a key is ever exposed, revoke it immediately at [console.groq.com/keys](https://console.groq.com/keys) and issue a new one.

---

## 📄 License
This project is submitted as a graduation project for the Data Science & AI Diploma. Free to adapt for educational and personal use.
