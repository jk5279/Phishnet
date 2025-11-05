"""
04_ml_pipeline.py

Machine Learning Pipeline

This script trains and evaluates multiple machine learning models for phishing detection:
1. Loads preprocessed ML dataset
2. Engineers metadata features
3. Creates train/test split
4. Trains multiple models (Logistic Regression, Linear SVC, SGD)
5. Performs cross-validation
6. Evaluates models with metrics
7. Hyperparameter tuning with grid search
8. Saves best model for inference

Output: models/ml_best_model.pkl and evaluation metrics
"""

import os
import string
import pickle
import json
import numpy as np
import pandas as pd
from collections import Counter
from time import time
from datetime import datetime
from typing import Tuple, Dict

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, learning_curve, GridSearchCV, train_test_split
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Configuration
# =========================

# Input/Output paths
INPUT_FILE = os.path.join("cleaned_data", "ml_dataset_final.csv")
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "ml_best_model.pkl")
LEARNING_CURVE_PATH = os.path.join(MODEL_DIR, "ml_learning_curve.png")
EVALUATION_LOG_PATH = os.path.join(MODEL_DIR, "ml_evaluation_log.json")
EVALUATION_REPORT_PATH = os.path.join(MODEL_DIR, "ml_evaluation_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "ml_confusion_matrix.png")

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# TF-IDF settings
MAX_FEATURES = 40000
NGRAM_RANGE = (1, 2)

# Grid search settings (simplified for faster execution)
GRID_SEARCH_PARAMS = {
    'preprocessor__text__max_features': [10000, 20000],
    'preprocessor__text__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1.0],
    'clf__class_weight': [None, 'balanced']
}


# =========================
# Data Loading and Preparation
# =========================

def load_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data for training."""
    print("\n--- 1. Loading and Preparing Data ---")
    df = pd.read_csv(filepath)
    print(f"Dataset size: {df.shape}")
    print(f"Label distribution: {Counter(df['label'])}")

    # Engineer metadata features
    print("Engineering metadata features...")
    X = df[['text']].copy()
    X['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    X['punct_count'] = df['text'].apply(
        lambda x: sum(1 for c in str(x) if c in string.punctuation)
    )
    X['upper_count'] = df['text'].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()])
    )
    y = df['label']

    return X, y


def create_train_test_split(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Create stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Check for duplicates/leakage
    dups_between = set(X_train['text']).intersection(set(X_test['text']))
    if len(dups_between) > 0:
        print(f"WARNING: {len(dups_between)} duplicate texts between train/test")
    else:
        print("No duplicate texts between train/test - good!")

    return X_train, X_test, y_train, y_test


# =========================
# Pipeline Creation
# =========================

def create_preprocessor() -> ColumnTransformer:
    """Create the preprocessor with TF-IDF and numeric scaling."""
    text_transformer = TfidfVectorizer(
        stop_words='english',
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES
    )
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'text'),
            ('numeric', numeric_transformer, ['word_count', 'punct_count', 'upper_count'])
        ],
        remainder='drop'
    )
    return preprocessor


def create_model_pipelines() -> dict:
    """Create multiple model pipelines."""
    print("\n--- 2. Defining Preprocessor ---")
    preprocessor = create_preprocessor()

    print("--- 3. Defining ML Pipelines ---")
    pipelines = {
        "Logistic Regression": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression(random_state=RANDOM_STATE, solver='liblinear', max_iter=1000))
        ]),
        "Linear SVC (SVM)": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('clf', LinearSVC(random_state=RANDOM_STATE, C=0.1, dual=False, max_iter=2000))
        ]),
        "SGD Classifier (SVM-like)": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('clf', SGDClassifier(
                random_state=RANDOM_STATE, max_iter=1000, tol=1e-3, loss='hinge'
            ))
        ])
    }
    return pipelines


# =========================
# Model Training and Evaluation
# =========================

def train_and_evaluate_model(
    name: str, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """Train and evaluate a single model."""
    print(f"\nTraining {name}...")
    start_time = time()

    # Train
    pipeline.fit(X_train, y_train)
    train_time = time() - start_time

    # Predict
    start_time = time()
    y_pred = pipeline.predict(X_test)
    predict_time = time() - start_time

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Train time: {train_time:.2f} seconds")
    print(f"Predict time: {predict_time:.2f} seconds")
    print(f"\nAccuracy for {name}: {accuracy*100:.2f}%")
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))

    return {
        'name': name,
        'pipeline': pipeline,
        'accuracy': accuracy,
        'train_time': train_time,
        'predict_time': predict_time,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': y_pred
    }


def cross_validate_model(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[float, float]:
    """Perform cross-validation."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1
    )
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    print(f"CV accuracy: {mean_score*100:.2f}% +/- {std_score*100:.2f}% (n={len(cv_scores)})")
    return mean_score, std_score


def plot_learning_curve(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, output_path: str
):
    """Plot learning curve to visualize overfitting."""
    print("\n--- Generating Learning Curve ---")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy', n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Train', linewidth=2)
    plt.plot(train_sizes, val_mean, 'o-', label='Validation', linewidth=2)
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title('Learning Curve', fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f" - Saved: {output_path}")


# =========================
# Hyperparameter Tuning
# =========================

def perform_grid_search(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple:
    """Perform grid search for hyperparameter tuning."""
    print("\n--- Performing Grid Search ---")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipeline, GRID_SEARCH_PARAMS, cv=cv,
        scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"Best CV score: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")

    return grid.best_estimator_, grid.best_score_, grid.best_params_


# =========================
# Model Saving
# =========================

def save_model(model: Pipeline, filepath: str):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n--- Model Saved ---")
    print(f"Saved to: {filepath}")


def save_evaluation_logs(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    best_params: Dict,
    cv_score: float,
    test_accuracy: float,
    classification_report_text: str,
    confusion_matrix_array: np.ndarray,
    log_path: str,
    report_path: str,
    cm_path: str
):
    """
    Save comprehensive evaluation logs including metrics, reports, and visualizations.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        best_params: Best hyperparameters
        cv_score: Cross-validation score
        test_accuracy: Test set accuracy
        classification_report_text: Text classification report
        confusion_matrix_array: Confusion matrix array
        log_path: Path to save JSON log
        report_path: Path to save text report
        cm_path: Path to save confusion matrix image
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Calculate additional metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Save JSON log with all metrics
    evaluation_log = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'best_hyperparameters': best_params,
        'metrics': {
            'cross_validation_score': float(cv_score),
            'test_accuracy': float(test_accuracy),
            'test_precision_weighted': float(precision),
            'test_recall_weighted': float(recall),
            'test_f1_weighted': float(f1)
        },
        'confusion_matrix': confusion_matrix_array.tolist(),
        'classification_report': classification_report_text
    }
    
    with open(log_path, 'w') as f:
        json.dump(evaluation_log, f, indent=2)
    print(f" - Saved evaluation log: {log_path}")
    
    # Save text report
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ML Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: {model_name}\n\n")
        f.write("Best Hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nCross-Validation Score: {cv_score:.4f}\n")
        f.write(f"\nTest Set Metrics:\n")
        f.write(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"  Precision (weighted): {precision:.4f}\n")
        f.write(f"  Recall (weighted): {recall:.4f}\n")
        f.write(f"  F1 Score (weighted): {f1:.4f}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report_text)
        f.write(f"\nConfusion Matrix:\n")
        f.write(str(confusion_matrix_array))
        f.write("\n")
    print(f" - Saved evaluation report: {report_path}")
    
    # Save confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_array,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Phishing', 'Phishing'],
        yticklabels=['Not Phishing', 'Phishing']
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f" - Saved confusion matrix: {cm_path}")


# =========================
# Main Execution
# =========================

def main():
    """Main execution function for ML pipeline."""
    print("=" * 60)
    print("Machine Learning Pipeline")
    print("=" * 60)

    # Load and prepare data
    X, y = load_data(INPUT_FILE)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    # Create pipelines
    pipelines = create_model_pipelines()

    # Train and evaluate all models
    print("\n--- 4. Training and Evaluating Models ---")
    results = {}
    for name, pipeline in pipelines.items():
        result = train_and_evaluate_model(
            name, pipeline, X_train, y_train, X_test, y_test
        )
        results[name] = result

    # Cross-validation on best performing model
    print("\n--- 5. Cross-Validation ---")
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\nPerforming cross-validation on {best_model_name}...")
    best_pipeline = results[best_model_name]['pipeline']
    cv_mean, cv_std = cross_validate_model(best_pipeline, X_train, y_train)

    # Learning curve
    plot_learning_curve(best_pipeline, X_train, y_train, LEARNING_CURVE_PATH)

    # Grid search for hyperparameter tuning
    print("\n--- 6. Hyperparameter Tuning ---")
    base_pipeline = create_model_pipelines()[best_model_name]
    best_tuned_model, best_cv_score, best_params = perform_grid_search(
        base_pipeline, X_train, y_train
    )

    # Evaluate tuned model on test set
    print("\n--- 7. Evaluating Tuned Model ---")
    y_pred_tuned = best_tuned_model.predict(X_test)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    report_tuned = classification_report(y_test, y_pred_tuned)
    
    print(f"Test accuracy (tuned): {accuracy_tuned*100:.2f}%")
    print(f"Classification report (tuned):")
    print(report_tuned)

    # Save best model
    print("\n--- 8. Saving Best Model ---")
    save_model(best_tuned_model, BEST_MODEL_PATH)
    
    # Save evaluation logs
    print("\n--- 9. Saving Evaluation Logs ---")
    save_evaluation_logs(
        y_test=y_test,
        y_pred=y_pred_tuned,
        model_name=best_model_name,
        best_params=best_params,
        cv_score=best_cv_score,
        test_accuracy=accuracy_tuned,
        classification_report_text=report_tuned,
        confusion_matrix_array=cm_tuned,
        log_path=EVALUATION_LOG_PATH,
        report_path=EVALUATION_REPORT_PATH,
        cm_path=CONFUSION_MATRIX_PATH
    )

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"Best model: {best_model_name}")
    print(f"Best CV score: {best_cv_score:.4f}")
    print(f"Test accuracy: {accuracy_tuned*100:.2f}%")
    print(f"Best parameters: {best_params}")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

