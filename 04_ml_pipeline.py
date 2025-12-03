"""
04_ml_pipeline.py

Machine Learning Pipeline

This script trains and evaluates multiple machine learning models for phishing detection:
1. Loads preprocessed ML dataset from cleaned_data/ML/train/, validation/, test/
2. Engineers metadata features
3. Trains multiple models (Naive Bayes, Logistic Regression, Linear SVC, SGD)
4. Performs cross-validation for each model
5. Evaluates models with metrics
6. Hyperparameter tuning with grid search for each model
7. Saves all models and logs to ml_methods/ directory structure

Output: ml_methods/{method_name}/model.pkl and ml_methods/{method_name}/logs/
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
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    GridSearchCV,
    train_test_split,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Configuration
# =========================

# Input/Output paths
TRAIN_INPUT = os.path.join("cleaned_data", "ML", "train", "train_split.csv")
VAL_INPUT = os.path.join("cleaned_data", "ML", "validation", "validation_split.csv")
TEST_INPUT = os.path.join("cleaned_data", "ML", "test", "test_split.csv")
MODEL_DIR = "ml_methods"

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# TF-IDF settings
MAX_FEATURES = 40000
NGRAM_RANGE = (1, 2)

# Grid search settings (model-specific)
LOGISTIC_REGRESSION_GRID_SEARCH_PARAMS = {
    "preprocessor__text__max_features": [10000, 20000],
    "preprocessor__text__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0],
    "clf__class_weight": [None, "balanced"],
}

LINEAR_SVC_GRID_SEARCH_PARAMS = {
    "preprocessor__text__max_features": [10000, 20000],
    "preprocessor__text__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0],
    "clf__class_weight": [None, "balanced"],
}

SGD_CLASSIFIER_GRID_SEARCH_PARAMS = {
    "preprocessor__text__max_features": [10000, 20000],
    "preprocessor__text__ngram_range": [(1, 1), (1, 2)],
    "clf__alpha": [0.0001, 0.001, 0.01],  # SGD uses alpha instead of C
    "clf__class_weight": [None, "balanced"],
}

# Naive Bayes grid search params
NAIVE_BAYES_GRID_SEARCH_PARAMS = {
    "preprocessor__text__max_features": [10000, 20000],
    "preprocessor__text__ngram_range": [(1, 1), (1, 2)],
    "clf__alpha": [0.1, 0.5, 1.0],
}


# =========================
# Data Loading and Preparation
# =========================


def load_data(
    train_path: str, val_path: str = None, test_path: str = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and prepare data for training from split files."""
    print("\n--- 1. Loading and Preparing Data ---")

    # Load train data
    train_df = pd.read_csv(train_path)
    print(f"Train dataset size: {train_df.shape}")
    print(f"Train label distribution: {Counter(train_df['label'])}")

    # Engineer metadata features for train
    print("Engineering metadata features for train set...")
    X_train = train_df[["text"]].copy()
    X_train["word_count"] = train_df["text"].apply(lambda x: len(str(x).split()))
    X_train["punct_count"] = train_df["text"].apply(
        lambda x: sum(1 for c in str(x) if c in string.punctuation)
    )
    X_train["upper_count"] = train_df["text"].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()])
    )
    y_train = train_df["label"]

    # Load validation data if provided
    if val_path and os.path.exists(val_path):
        val_df = pd.read_csv(val_path)
        print(f"Validation dataset size: {val_df.shape}")
        X_val = val_df[["text"]].copy()
        X_val["word_count"] = val_df["text"].apply(lambda x: len(str(x).split()))
        X_val["punct_count"] = val_df["text"].apply(
            lambda x: sum(1 for c in str(x) if c in string.punctuation)
        )
        X_val["upper_count"] = val_df["text"].apply(
            lambda x: len([w for w in str(x).split() if w.isupper()])
        )
        y_val = val_df["label"]
    else:
        X_val = pd.DataFrame()
        y_val = pd.Series()

    # Load test data if provided
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"Test dataset size: {test_df.shape}")
        X_test = test_df[["text"]].copy()
        X_test["word_count"] = test_df["text"].apply(lambda x: len(str(x).split()))
        X_test["punct_count"] = test_df["text"].apply(
            lambda x: sum(1 for c in str(x) if c in string.punctuation)
        )
        X_test["upper_count"] = test_df["text"].apply(
            lambda x: len([w for w in str(x).split() if w.isupper()])
        )
        y_test = test_df["label"]
    else:
        X_test = pd.DataFrame()
        y_test = pd.Series()

    # Check for duplicates/leakage between train and test
    if not X_test.empty:
        dups_between = set(X_train["text"]).intersection(set(X_test["text"]))
        if len(dups_between) > 0:
            print(f"WARNING: {len(dups_between)} duplicate texts between train/test")
        else:
            print("No duplicate texts between train/test - good!")

    return X_train, y_train, X_val, y_val, X_test, y_test


# =========================
# Pipeline Creation
# =========================


def create_preprocessor(use_minmax=False) -> ColumnTransformer:
    """Create the preprocessor with TF-IDF and numeric scaling."""
    text_transformer = TfidfVectorizer(
        stop_words="english", ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES
    )
    # Use MinMaxScaler for Naive Bayes (non-negative), StandardScaler for others
    numeric_transformer = MinMaxScaler() if use_minmax else StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, "text"),
            (
                "numeric",
                numeric_transformer,
                ["word_count", "punct_count", "upper_count"],
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def create_model_pipelines() -> dict:
    """Create multiple model pipelines."""
    print("\n--- 2. Defining Preprocessor ---")
    # Create preprocessor with MinMaxScaler for Naive Bayes (non-negative values required)
    preprocessor_nb = create_preprocessor(use_minmax=True)
    # Create preprocessor with StandardScaler for other models
    preprocessor_standard = create_preprocessor(use_minmax=False)

    print("--- 3. Defining ML Pipelines ---")
    pipelines = {
        "Naive Bayes (Baseline)": Pipeline(
            steps=[("preprocessor", preprocessor_nb), ("clf", MultinomialNB(alpha=1.0))]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor_standard),
                (
                    "clf",
                    LogisticRegression(
                        random_state=RANDOM_STATE, solver="liblinear", max_iter=1000
                    ),
                ),
            ]
        ),
        "Linear SVC (SVM)": Pipeline(
            steps=[
                ("preprocessor", preprocessor_standard),
                (
                    "clf",
                    LinearSVC(
                        random_state=RANDOM_STATE, C=0.1, dual=False, max_iter=2000
                    ),
                ),
            ]
        ),
        "SGD Classifier (SVM-like)": Pipeline(
            steps=[
                ("preprocessor", preprocessor_standard),
                (
                    "clf",
                    SGDClassifier(
                        random_state=RANDOM_STATE, max_iter=1000, tol=1e-3, loss="hinge"
                    ),
                ),
            ]
        ),
    }
    return pipelines


# =========================
# Model Training and Evaluation
# =========================


def train_and_evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
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
        "name": name,
        "pipeline": pipeline,
        "accuracy": accuracy,
        "train_time": train_time,
        "predict_time": predict_time,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }


def cross_validate_model(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[float, float]:
    """Perform cross-validation."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    print(
        f"CV accuracy: {mean_score*100:.2f}% +/- {std_score*100:.2f}% (n={len(cv_scores)})"
    )
    return mean_score, std_score


def plot_learning_curve(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, model_name: str
):
    """Plot learning curve to visualize overfitting."""
    print("\n--- Generating Learning Curve ---")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, "o-", label="Train", linewidth=2)
    plt.plot(train_sizes, val_mean, "o-", label="Validation", linewidth=2)
    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title(f"Learning Curve - {model_name}", fontsize=14)
    plt.tight_layout()

    paths = get_method_paths(model_name)
    os.makedirs(paths["logs_dir"], exist_ok=True)
    plt.savefig(paths["learning_curve_path"], dpi=150)
    plt.close()
    print(f" - Saved: {paths['learning_curve_path']}")


# =========================
# Hyperparameter Tuning
# =========================


def perform_grid_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = None,
) -> tuple:
    """Perform grid search for hyperparameter tuning."""
    print("\n--- Performing Grid Search ---")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Select appropriate grid search parameters based on model name
    if model_name and "Naive Bayes" in model_name:
        search_params = NAIVE_BAYES_GRID_SEARCH_PARAMS
    elif model_name and "Logistic Regression" in model_name:
        search_params = LOGISTIC_REGRESSION_GRID_SEARCH_PARAMS
    elif model_name and "Linear SVC" in model_name:
        search_params = LINEAR_SVC_GRID_SEARCH_PARAMS
    elif model_name and "SGD" in model_name:
        search_params = SGD_CLASSIFIER_GRID_SEARCH_PARAMS
    else:
        # Default to Logistic Regression params if model name not recognized
        search_params = LOGISTIC_REGRESSION_GRID_SEARCH_PARAMS
        print(f"Warning: Unknown model name '{model_name}', using default grid search params")

    grid = GridSearchCV(
        pipeline, search_params, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"Best CV score: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")

    return grid.best_estimator_, grid.best_score_, grid.best_params_


# =========================
# Path Helpers
# =========================


def get_method_dir_name(model_name: str) -> str:
    """Convert model name to directory name."""
    name_mapping = {
        "Naive Bayes (Baseline)": "naive_bayes",
        "Logistic Regression": "logistic_regression",
        "Linear SVC (SVM)": "linear_svc",
        "SGD Classifier (SVM-like)": "sgd_classifier",
    }
    return name_mapping.get(model_name, model_name.lower().replace(" ", "_"))


def get_method_paths(model_name: str) -> Dict[str, str]:
    """Get all paths for a specific method."""
    method_dir = get_method_dir_name(model_name)
    method_base_dir = os.path.join(MODEL_DIR, method_dir)
    logs_dir = os.path.join(method_base_dir, "logs")

    return {
        "method_dir": method_base_dir,
        "logs_dir": logs_dir,
        "model_path": os.path.join(method_base_dir, "model.pkl"),
        "learning_curve_path": os.path.join(logs_dir, "learning_curve.png"),
        "evaluation_log_path": os.path.join(logs_dir, "evaluation_log.json"),
        "evaluation_report_path": os.path.join(logs_dir, "evaluation_report.txt"),
        "confusion_matrix_path": os.path.join(logs_dir, "confusion_matrix.png"),
    }


# =========================
# Model Saving
# =========================


def save_model(model: Pipeline, model_name: str):
    """Save trained model to method-specific directory."""
    paths = get_method_paths(model_name)
    os.makedirs(paths["method_dir"], exist_ok=True)
    with open(paths["model_path"], "wb") as f:
        pickle.dump(model, f)
    print(f"\n--- Model Saved ---")
    print(f"Saved to: {paths['model_path']}")


def save_evaluation_logs(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    best_params: Dict,
    cv_score: float,
    test_accuracy: float,
    classification_report_text: str,
    confusion_matrix_array: np.ndarray,
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
    """
    paths = get_method_paths(model_name)
    os.makedirs(paths["logs_dir"], exist_ok=True)

    # Calculate additional metrics
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Save JSON log with all metrics
    evaluation_log = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "best_hyperparameters": best_params,
        "metrics": {
            "cross_validation_score": float(cv_score),
            "test_accuracy": float(test_accuracy),
            "test_precision_weighted": float(precision),
            "test_recall_weighted": float(recall),
            "test_f1_weighted": float(f1),
        },
        "confusion_matrix": confusion_matrix_array.tolist(),
        "classification_report": classification_report_text,
    }

    with open(paths["evaluation_log_path"], "w") as f:
        json.dump(evaluation_log, f, indent=2)
    print(f" - Saved evaluation log: {paths['evaluation_log_path']}")

    # Save text report
    with open(paths["evaluation_report_path"], "w") as f:
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
    print(f" - Saved evaluation report: {paths['evaluation_report_path']}")

    # Save confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Phishing", "Phishing"],
        yticklabels=["Not Phishing", "Phishing"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(paths["confusion_matrix_path"], dpi=150)
    plt.close()
    print(f" - Saved confusion matrix: {paths['confusion_matrix_path']}")


# =========================
# Main Execution
# =========================


def main():
    """Main execution function for ML pipeline."""
    print("=" * 60)
    print("Machine Learning Pipeline")
    print("=" * 60)

    # Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        TRAIN_INPUT, VAL_INPUT, TEST_INPUT
    )

    if X_test.empty:
        print("WARNING: Test set not found. Using validation set for evaluation.")
        X_test, y_test = X_val, y_val

    # Create pipelines
    pipelines = create_model_pipelines()

    # Train and evaluate all models
    print("\n--- 4. Training and Evaluating All Models ---")
    all_results = {}

    for name, pipeline in pipelines.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {name}")
        print(f"{'=' * 60}")

        # Initial training and evaluation
        result = train_and_evaluate_model(
            name, pipeline, X_train, y_train, X_test, y_test
        )
        all_results[name] = result

        # Cross-validation
        print(f"\n--- Cross-Validation for {name} ---")
        cv_mean, cv_std = cross_validate_model(pipeline, X_train, y_train)
        result["cv_mean"] = cv_mean
        result["cv_std"] = cv_std

        # Learning curve
        plot_learning_curve(pipeline, X_train, y_train, name)

        # Grid search for hyperparameter tuning
        print(f"\n--- Hyperparameter Tuning for {name} ---")
        base_pipeline = create_model_pipelines()[name]
        tuned_model, tuned_cv_score, tuned_params = perform_grid_search(
            base_pipeline, X_train, y_train, name
        )

        # Evaluate tuned model on test set
        print(f"\n--- Evaluating Tuned {name} ---")
        y_pred_tuned = tuned_model.predict(X_test)
        accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
        cm_tuned = confusion_matrix(y_test, y_pred_tuned)
        report_tuned = classification_report(y_test, y_pred_tuned)

        print(f"Test accuracy (tuned): {accuracy_tuned*100:.2f}%")
        print(f"Classification report (tuned):")
        print(report_tuned)

        # Save model and logs
        print(f"\n--- Saving {name} ---")
        save_model(tuned_model, name)
        save_evaluation_logs(
            y_test=y_test,
            y_pred=y_pred_tuned,
            model_name=name,
            best_params=tuned_params,
            cv_score=tuned_cv_score,
            test_accuracy=accuracy_tuned,
            classification_report_text=report_tuned,
            confusion_matrix_array=cm_tuned,
        )

        # Update results with tuned metrics
        all_results[name]["tuned_accuracy"] = accuracy_tuned
        all_results[name]["tuned_cv_score"] = tuned_cv_score
        all_results[name]["tuned_params"] = tuned_params
        all_results[name]["tuned_model"] = tuned_model

    # Report all models equally
    print("\n" + "=" * 60)
    print("All Models Summary")
    print("=" * 60)
    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  Test Accuracy: {result['tuned_accuracy']*100:.2f}%")
        print(f"  CV Score: {result['tuned_cv_score']:.4f}")
        print(f"  Best Params: {result['tuned_params']}")
        paths = get_method_paths(name)
        print(f"  Saved to: {paths['model_path']}")

    # Identify and report best model
    best_model_name = max(all_results, key=lambda k: all_results[k]["tuned_accuracy"])
    best_result = all_results[best_model_name]

    print("\n" + "=" * 60)
    print("Best Model")
    print("=" * 60)
    print(f"Model: {best_model_name}")
    print(f"Test Accuracy: {best_result['tuned_accuracy']*100:.2f}%")
    print(f"CV Score: {best_result['tuned_cv_score']:.4f}")
    print(f"Best Parameters: {best_result['tuned_params']}")
    best_paths = get_method_paths(best_model_name)
    print(f"Model saved to: {best_paths['model_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
