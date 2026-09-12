# %% [markdown]
# # College Chatbot — Data Preprocessing, EDA & Model Training
#
# This script follows the 8 required preprocessing steps for the graduation
# project, end to end, on the "College Subjects & FAQ Chatbot" intent
# dataset, then trains and evaluates an intent-classification model that
# powers PyBot's fast FAQ-answer path.

# %%
import os
import re
import string
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "college_chatbot_dataset.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORT_PATH = os.path.join(BASE_DIR, "outputs", "REPORT.md")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

report_lines = ["# College Chatbot — Data & Model Report\n"]


def log(line=""):
    print(line)
    report_lines.append(line)


# %% [markdown]
# ## Step 1 — Load and Explore the Dataset

# %%
df = pd.read_csv(DATA_PATH)

log("## Step 1: Load and Explore the Dataset\n")
log(f"- Shape: {df.shape}")
log(f"- Columns: {list(df.columns)}")
log(f"- Dtypes: {df.dtypes.to_dict()}")
log(f"- Number of unique intents: {df['intent'].nunique()}")
log("")
log("Sample rows:\n")
log(df.sample(5, random_state=42).to_markdown(index=False))
log("")

# %% [markdown]
# ## Step 2 — Check for and Handle Duplicates

# %%
n_dupes = df.duplicated(subset=["text"]).sum()
log("## Step 2: Check for and Handle Duplicates\n")
log(f"- Duplicate rows found (by text): {n_dupes} "
    f"({n_dupes / len(df):.1%} of dataset)")

df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
log(f"- Rows after dropping duplicates: {len(df)}")
log("")

# %% [markdown]
# ## Step 3 — Apply Techniques to Handle Missing Values

# %%
missing = df.isna().sum()
log("## Step 3: Apply Techniques to Handle Missing Values\n")
log("Missing values per column:\n")
log(missing.to_frame("missing_count").to_markdown())

if missing.sum() > 0:
    df = df.dropna(subset=["text", "intent"]).reset_index(drop=True)
    log(f"- Dropped rows with missing text/intent. New shape: {df.shape}")
else:
    log("- No missing values found — no action needed.")
log("")


# %% [markdown]
# ## Text cleaning (needed before outlier / NLP steps)

# %%
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df["text"].apply(clean_text)
df["word_count"] = df["clean_text"].apply(lambda t: len(t.split()))

# %% [markdown]
# ## Step 4 — Check Outliers and Handle Them
# (Outlier definition here: unusually short/long messages by word count.)

# %%
q1, q3 = df["word_count"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(df["word_count"] < lower) | (df["word_count"] > upper)]

log("## Step 4: Check Outliers and Handle Them\n")
log(f"- Word count IQR bounds: [{lower:.1f}, {upper:.1f}]")
log(f"- Outlier messages found: {len(outliers)} ({len(outliers) / len(df):.1%})")

plt.figure(figsize=(6, 4))
sns.boxplot(x=df["word_count"], color="#1E2761")
plt.title("Message Length (word count) — Outlier Check")
plt.xlabel("Word count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_outlier_boxplot.png"), dpi=150)
plt.close()

# Cap extreme outliers instead of dropping (keep <5% affected as per project rule)
if len(outliers) / len(df) < 0.05:
    log("- Outlier share is below 5% — retained as-is (per project guideline).")
else:
    cap = int(upper)
    df["clean_text"] = df.apply(
        lambda r: " ".join(r["clean_text"].split()[:cap])
        if r["word_count"] > upper
        else r["clean_text"],
        axis=1,
    )
    log(f"- Long outliers truncated to {cap} words.")
log("")

# %% [markdown]
# ## Step 5 — Make Visualizations to Check Problems, Distribution, and Balance

# %%
log("## Step 5: Visualizations\n")

# 1) Class distribution
plt.figure(figsize=(9, 5))
order = df["intent"].value_counts().index
sns.countplot(y="intent", data=df, order=order, color="#1E2761")
plt.title("Intent Class Distribution")
plt.xlabel("Count")
plt.ylabel("Intent")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_class_distribution.png"), dpi=150)
plt.close()

# 2) Word count histogram
plt.figure(figsize=(6, 4))
sns.histplot(df["word_count"], bins=12, color="#3B4A9E", kde=True)
plt.title("Distribution of Message Length (word count)")
plt.xlabel("Word count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_word_count_hist.png"), dpi=150)
plt.close()

# 3) Top words overall (bar chart, in place of a word cloud)
all_words = " ".join(df["clean_text"]).split()
stop = set(
    "a an the is are was were be been being to of in on for and or with "
    "what how do i my me you your it this that does can are's".split()
)
freq = pd.Series([w for w in all_words if w not in stop]).value_counts().head(20)
plt.figure(figsize=(7, 6))
sns.barplot(x=freq.values, y=freq.index, color="#028090")
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_top_words.png"), dpi=150)
plt.close()

# 4) Missing/duplicate/outlier summary heatmap-style bar
issues = pd.Series(
    {"duplicates_removed": n_dupes, "missing_values": int(missing.sum()), "outliers": len(outliers)}
)
plt.figure(figsize=(5, 4))
sns.barplot(x=issues.index, y=issues.values, color="#990011")
plt.title("Data Quality Issues Found")
plt.ylabel("Count")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_data_quality_summary.png"), dpi=150)
plt.close()

log("- Saved: class distribution, word-count histogram, top-words bar chart, "
    "data-quality summary (see outputs/plots/).")
log("")

# %% [markdown]
# ## Step 6 — Check Imbalance and Handle It

# %%
counts = df["intent"].value_counts()
min_ratio = counts.min() / counts.max()
log("## Step 6: Check Imbalance and Handle It\n")
log(f"- Smallest class: {counts.idxmin()} ({counts.min()} examples)")
log(f"- Largest class: {counts.idxmax()} ({counts.max()} examples)")
log(f"- Minority/majority ratio: {min_ratio:.2f} "
    f"({'IMBALANCED' if min_ratio < 0.70 else 'balanced'} — threshold 0.70)")

# Balance via random oversampling of minority classes up to the max class size
target = counts.max()
balanced_parts = []
for intent, group in df.groupby("intent"):
    if len(group) < target:
        extra = group.sample(target - len(group), replace=True, random_state=42)
        balanced_parts.append(pd.concat([group, extra]))
    else:
        balanced_parts.append(group)
df_balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)

plt.figure(figsize=(9, 5))
order2 = df_balanced["intent"].value_counts().index
sns.countplot(y="intent", data=df_balanced, order=order2, color="#02C39A")
plt.title("Intent Class Distribution — After Oversampling")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_class_distribution_balanced.png"), dpi=150)
plt.close()

log(f"- Applied random oversampling (with replacement) so every class has "
    f"{target} examples — new dataset size: {len(df_balanced)} rows.")
log("- Before/after class distribution saved to outputs/plots/.")
log("")

# %% [markdown]
# ## Feature Extraction (TF-IDF) + Train/Test Split

# %%
X = df_balanced["clean_text"]
y = df_balanced["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

log("## Feature Extraction\n")
log(f"- TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
log(f"- Train size: {X_train_vec.shape[0]}, Test size: {X_test_vec.shape[0]}")
log("")

# %% [markdown]
# ## Step 7 — Address Overfitting and Underfitting
# Baseline model (Multinomial Naive Bayes) vs. tuned final model (Logistic
# Regression, grid-searched), compared via cross-validation and a learning
# curve.

# %%
log("## Step 7: Address Overfitting and Underfitting\n")

baseline = MultinomialNB()
baseline.fit(X_train_vec, y_train)
baseline_train_acc = accuracy_score(y_train, baseline.predict(X_train_vec))
baseline_test_acc = accuracy_score(y_test, baseline.predict(X_test_vec))
log(f"- Baseline (Naive Bayes): train acc = {baseline_train_acc:.3f}, "
    f"test acc = {baseline_test_acc:.3f}")

param_grid = {"C": [0.1, 0.5, 1, 2, 5, 10]}
grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1_macro",
)
grid.fit(X_train_vec, y_train)
final_model = grid.best_estimator_
log(f"- Grid search best C: {grid.best_params_['C']}")

final_train_acc = accuracy_score(y_train, final_model.predict(X_train_vec))
final_test_acc = accuracy_score(y_test, final_model.predict(X_test_vec))
log(f"- Final (Logistic Regression): train acc = {final_train_acc:.3f}, "
    f"test acc = {final_test_acc:.3f}")

cv_scores = cross_val_score(
    final_model, X_train_vec, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
)
log(f"- 5-fold cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

gap = final_train_acc - final_test_acc
if gap > 0.15:
    verdict = "Some overfitting detected (train/test gap > 0.15) — consider more data or stronger regularization."
elif final_test_acc < 0.6:
    verdict = "Underfitting risk — test accuracy is low, consider richer features or more data."
else:
    verdict = "Train/test gap is small and CV accuracy is stable — the model generalizes well."
log(f"- Verdict: {verdict}")

# Learning curve
train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(max_iter=1000, C=grid.best_params_["C"]),
    X_train_vec, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    train_sizes=np.linspace(0.2, 1.0, 5),
    scoring="accuracy",
)
plt.figure(figsize=(6, 4))
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#1E2761", label="Training score")
plt.plot(train_sizes, val_scores.mean(axis=1), "o-", color="#02C39A", label="Validation score")
plt.title("Learning Curve — Logistic Regression")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_learning_curve.png"), dpi=150)
plt.close()
log("- Learning curve saved to outputs/plots/07_learning_curve.png")
log("")

# %% [markdown]
# ## Step 8 — Evaluate Model Performance Using Metrics

# %%
y_pred = final_model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

log("## Step 8: Evaluate Model Performance Using Metrics\n")
log(f"- Accuracy: {acc:.3f}")
log(f"- Precision (macro): {prec:.3f}")
log(f"- Recall (macro): {rec:.3f}")
log(f"- F1-score (macro): {f1:.3f}")
log("")
log("Classification report:\n")
log("```\n" + classification_report(y_test, y_pred, zero_division=0) + "\n```")

fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, ax=ax, xticks_rotation=90, colorbar=False, cmap="Blues"
)
plt.title("Confusion Matrix — Final Model")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_confusion_matrix.png"), dpi=150)
plt.close()

log("- Confusion matrix saved to outputs/plots/08_confusion_matrix.png")
log("")
log(f"**Baseline vs. Final model — test accuracy:** "
    f"{baseline_test_acc:.3f} (Naive Bayes) → {final_test_acc:.3f} (Logistic Regression)")

# %% [markdown]
# ## Save Model Artifacts

# %%
joblib.dump(final_model, os.path.join(MODEL_DIR, "intent_classifier.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\nSaved model + vectorizer to:", MODEL_DIR)
print("Saved full report to:", REPORT_PATH)
