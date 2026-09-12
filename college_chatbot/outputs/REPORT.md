# College Chatbot — Data & Model Report

## Step 1: Load and Explore the Dataset

- Shape: (221, 2)
- Columns: ['text', 'intent']
- Dtypes: {'text': <StringDtype(storage='python', na_value=nan)>, 'intent': <StringDtype(storage='python', na_value=nan)>}
- Number of unique intents: 16

Sample rows:

| text                                            | intent              |
|:------------------------------------------------|:--------------------|
| What is sampling in statistics?                 | statistics          |
| How many exams are left this term?              | exam_schedule       |
| What is a database schema?                      | databases           |
| Can I register for a course after the deadline? | course_registration |
| Bye                                             | goodbye             |

## Step 2: Check for and Handle Duplicates

- Duplicate rows found (by text): 0 (0.0% of dataset)
- Rows after dropping duplicates: 221

## Step 3: Apply Techniques to Handle Missing Values

Missing values per column:

|        |   missing_count |
|:-------|----------------:|
| text   |               0 |
| intent |               0 |
- No missing values found — no action needed.

## Step 4: Check Outliers and Handle Them

- Word count IQR bounds: [1.0, 9.0]
- Outlier messages found: 6 (2.7%)
- Outlier share is below 5% — retained as-is (per project guideline).

## Step 5: Visualizations

- Saved: class distribution, word-count histogram, top-words bar chart, data-quality summary (see outputs/plots/).

## Step 6: Check Imbalance and Handle It

- Smallest class: library_hours (8 examples)
- Largest class: python_basics (22 examples)
- Minority/majority ratio: 0.36 (IMBALANCED — threshold 0.70)
- Applied random oversampling (with replacement) so every class has 22 examples — new dataset size: 352 rows.
- Before/after class distribution saved to outputs/plots/.

## Feature Extraction

- TF-IDF vocabulary size: 495
- Train size: 281, Test size: 71

## Step 7: Address Overfitting and Underfitting

- Baseline (Naive Bayes): train acc = 0.993, test acc = 0.789
- Grid search best C: 5
- Final (Logistic Regression): train acc = 0.993, test acc = 0.789
- 5-fold cross-validation accuracy: 0.872 ± 0.042
- Verdict: Some overfitting detected (train/test gap > 0.15) — consider more data or stronger regularization.
- Learning curve saved to outputs/plots/07_learning_curve.png

## Step 8: Evaluate Model Performance Using Metrics

- Accuracy: 0.789
- Precision (macro): 0.878
- Recall (macro): 0.787
- F1-score (macro): 0.809

Classification report:

```
                     precision    recall  f1-score   support

assignment_deadline       0.80      1.00      0.89         4
course_registration       1.00      1.00      1.00         4
    data_structures       0.50      0.25      0.33         4
          databases       0.80      0.80      0.80         5
      exam_schedule       1.00      0.60      0.75         5
       general_help       1.00      0.75      0.86         4
            goodbye       1.00      0.80      0.89         5
     grading_system       1.00      0.80      0.89         5
           greeting       1.00      1.00      1.00         4
      library_hours       1.00      1.00      1.00         4
   machine_learning       0.29      0.80      0.42         5
       office_hours       1.00      0.75      0.86         4
      python_basics       0.67      0.80      0.73         5
         statistics       1.00      0.50      0.67         4
  technical_support       1.00      0.75      0.86         4
             thanks       1.00      1.00      1.00         5

           accuracy                           0.79        71
          macro avg       0.88      0.79      0.81        71
       weighted avg       0.87      0.79      0.81        71

```
- Confusion matrix saved to outputs/plots/08_confusion_matrix.png

**Baseline vs. Final model — test accuracy:** 0.789 (Naive Bayes) → 0.789 (Logistic Regression)