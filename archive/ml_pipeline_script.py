# python
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

# Assumes you still load data from `ml_dataset_final.csv`
df = pd.read_csv("cleaned_data/ml_dataset_final.csv")
print("Dataset size:", df.shape)
print("Label distribution:", Counter(df['label']))

# Recreate train/test split as in your notebook
from sklearn.model_selection import train_test_split
X = df[['text']].copy()
X['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
X['punct_count'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c in __import__('string').punctuation))
X['upper_count'] = df['text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check duplicates / leakage
dups_between = set(X_train['text']).intersection(set(X_test['text']))
print("Duplicate texts between train/test:", len(dups_between))

# Recreate your pipeline (copy the one from your notebook; adjust names if needed)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=40000)
numeric_transformer = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'text'),
        ('numeric', numeric_transformer, ['word_count', 'punct_count', 'upper_count'])
    ],
    remainder='drop'
)
pipeline = Pipeline([('preprocessor', preprocessor),
                     ('clf', LogisticRegression(solver='liblinear', random_state=42))])

# Cross-validation to get a reliable estimate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"CV accuracy: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%  (n={len(cv_scores)})")

# Train on full train set and evaluate on test set to reproduce the reported gap
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Classification report on test set:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Learning curve (visualize under/overfitting)
train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X_train, y_train, cv=cv, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy', n_jobs=-1
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(6,4))
plt.plot(train_sizes, train_mean, 'o-', label='train')
plt.plot(train_sizes, val_mean, 'o-', label='val')
plt.xlabel('Train size'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.title('Learning Curve')
plt.show()

# Simple Grid Search to reduce overfitting: tune TF-IDF max_features, ngram_range and regularization C
param_grid = {
    'preprocessor__text__max_features': [5000, 10000, 20000],
    'preprocessor__text__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.01, 0.1, 1.0],
    'clf__class_weight': [None, 'balanced']
}
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print("Best CV score:", grid.best_score_)
print("Best params:", grid.best_params_)

# Evaluate best estimator on test set
best = grid.best_estimator_
y_pred_best = best.predict(X_test)
print("Test accuracy (best):", accuracy_score(y_test, y_pred_best))
print("Classification report (best):\n", classification_report(y_test, y_pred_best))

sample = [
    # Example 1: Clear phishing
"""Tuition Payment Deadline Missed – Final Grace Period Mohit Malhotra<mohit.malhotra@mail.utoronto.ca> Dear Valued Student, This is your final warning. As of October 6, 2025, our records indicate that your Fall 2025 tuition deposit remains unpaid. You are now outside the formal payment deadline of September 30, 2025, and are at immediate risk of deregistration, loss of student status, and deactivation of all university services.Under the University of Toronto’s Financial Registration Policy, a minimum tuition deposit of $3,000–$5,000 is mandatory for all students, regardless of OSAP or external funding status. OSAP and other financial aid sources are disbursed after the term begins and do not replace the required initial deposit. To retain your course enrolment and active student status, you have been granted a final short-term grace period. Payment must now be submitted immediately using the instructions below: 🔹 Interac e-Transfer Details Email: Mabintydumbuya_19@hotmail.com Memo: Include your student number onlySecurity Question: Your name
Answer: Your student ID number⚠️ Do not send receipts to the transfer email address (Mabintydumbuya_19@hotmail.com)
All receipts and communications must be submitted via this official university correspondence channel only including the security questions and answers attached to your payment.
Failure to comply by the grace period will result in the following irreversible actions:
Immediate removal from all enrolled courses
Full deactivation of UTmail+, ACORN, Quercus, and all student systems
Placement of academic and financial holds, blocking transcript access, graduation eligibility, and future enrolment
Deregistration from the University of Toronto, with permanent implications for your academic record and immigration/visa status (if applicable)
If you are experiencing a genuine financial emergency, you must respond today with a detailed explanation and a formal request for urgent financial assistance. Failure to respond will be treated as non-compliance.
To resolve your status today, reply to this email with one of the following:
A copy of your Interac e-Transfer confirmation/receipt
A brief explanation for your non-payment and expected payment date
A formal request for emergency financial or legal aid
We cannot and will not hold your seat in courses without immediate compliance. This is your final opportunity to secure your enrolment and student status.
Best regards,
Mohit Malhotra
Office of the Bursar
Financial Aid & Awards University of Toronto""",
]

best = grid.best_estimator_
y_pred = best.predict(pd.DataFrame({'text': sample}))
print("Prediction for sample:", y_pred)
