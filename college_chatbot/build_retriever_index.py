"""
build_retriever_index.py
--------------------------
Builds the retrieval index for the RAG pipeline, over the knowledge base
chunks in data/knowledge_base.json.

Default retriever: TF-IDF + cosine similarity (pure scikit-learn, runs
anywhere, no GPU or internet needed).

Optional upgrade: if you generate real sentence-embeddings in Google Colab
(see colab_generate_embeddings.ipynb) and drop the resulting
kb_embeddings.npy file into model/, the app will automatically prefer
those dense embeddings over TF-IDF for better semantic retrieval.
"""

import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "data", "knowledge_base.json")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

with open(KB_PATH, "r", encoding="utf-8") as f:
    kb = json.load(f)

texts = [f"{c['title']}. {c['text']}" for c in kb]

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
matrix = vectorizer.fit_transform(texts)

joblib.dump(vectorizer, os.path.join(MODEL_DIR, "kb_tfidf_vectorizer.pkl"))
joblib.dump(matrix, os.path.join(MODEL_DIR, "kb_tfidf_matrix.pkl"))
joblib.dump(kb, os.path.join(MODEL_DIR, "kb_chunks.pkl"))

print(f"Indexed {len(kb)} knowledge-base chunks.")
print(f"TF-IDF matrix shape: {matrix.shape}")
print("Saved: kb_tfidf_vectorizer.pkl, kb_tfidf_matrix.pkl, kb_chunks.pkl")
