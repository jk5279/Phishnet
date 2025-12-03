"""
Model Results Aggregation Script

Collects and aggregates final test results from all ML and DL models,
generating a comprehensive CSV report with accuracy, weighted precision,
and parameter counts.
"""

import os
import json
import pickle
import glob
import pandas as pd
from typing import Dict, List, Optional, Tuple
import torch
from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    AutoModelForSequenceClassification
)

# =========================
# Configuration
# =========================

ML_METHODS_DIR = "ml_methods"
DL_METHODS_DIR = "dl_methods"
OUTPUT_CSV = "model_results_summary.csv"

# Known parameter estimates (fallback)
KNOWN_PARAMETER_ESTIMATES = {
    "BERT": 110_000_000,
    "RoBERTa": 125_000_000,
    "DistilBERT": 66_000_000,
    "Logistic Regression": 500_000,  # Approximate: TF-IDF features + weights
    "Linear SVC (SVM)": 500_000,  # Similar to Logistic Regression
    "SGD Classifier (SVM-like)": 500_000,  # Similar to Logistic Regression
    "Naive Bayes (Baseline)": 200_000,  # Smaller feature set
    "Phishscore": 500_000,  # Unknown, estimate similar to other ML models
}

# =========================
# Parameter Counting Functions
# =========================

def count_ml_parameters(model_path: str, model_name: str) -> Optional[int]:
    """Try to count parameters from an ML model file."""
    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
        
        # Extract the classifier from the pipeline
        classifier = pipeline.named_steps.get('clf')
        if classifier is None:
            return None
        
        # Count parameters based on classifier type
        param_count = 0
        
        # For sklearn models, estimate based on feature count and model type
        if hasattr(classifier, 'coef_'):
            # Linear models (Logistic Regression, LinearSVC, SGD)
            if classifier.coef_ is not None:
                # coef_ shape: (n_classes, n_features)
                coef_params = classifier.coef_.size
                if hasattr(classifier, 'intercept_') and classifier.intercept_ is not None:
                    intercept_params = classifier.intercept_.size
                    param_count = coef_params + intercept_params
                else:
                    param_count = coef_params
        
        elif hasattr(classifier, 'feature_count_'):
            # Naive Bayes
            if classifier.feature_count_ is not None:
                param_count = classifier.feature_count_.size
            if hasattr(classifier, 'class_count_') and classifier.class_count_ is not None:
                param_count += classifier.class_count_.size
        
        # If we couldn't count, return None to use estimate
        if param_count == 0:
            return None
        
        return int(param_count)
    
    except Exception as e:
        print(f"Warning: Could not count parameters for {model_name}: {e}")
        return None

def count_dl_parameters(model_path: str, model_name: str) -> Optional[int]:
    """Try to count parameters from a DL model checkpoint."""
    try:
        # Try loading from best model first, then final
        best_path = os.path.join(model_path, "best")
        final_path = os.path.join(model_path, "final")
        
        model_dir = None
        if os.path.exists(best_path):
            model_dir = best_path
        elif os.path.exists(final_path):
            model_dir = final_path
        else:
            return None
        
        # Load model using AutoModel (works for all transformer types)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True
        )
        
        # Count all parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return int(total_params)
    
    except Exception as e:
        print(f"Warning: Could not count parameters for {model_name}: {e}")
        return None

def format_parameters(num_params: int) -> str:
    """Format parameter count in human-readable format."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    else:
        return str(num_params)

# =========================
# Data Collection Functions
# =========================

def collect_ml_results() -> List[Dict]:
    """Collect results from all ML models."""
    results = []
    
    # Find all ML model directories
    ml_dirs = glob.glob(os.path.join(ML_METHODS_DIR, "*"))
    
    for ml_dir in ml_dirs:
        if not os.path.isdir(ml_dir):
            continue
        
        model_dir_name = os.path.basename(ml_dir)
        log_path = os.path.join(ml_dir, "logs", "evaluation_log.json")
        model_path = os.path.join(ml_dir, "model.pkl")
        
        if not os.path.exists(log_path):
            print(f"Warning: No evaluation log found for {model_dir_name}")
            continue
        
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            model_name = log_data.get("model_name", model_dir_name)
            
            # Extract metrics
            metrics = log_data.get("metrics", {})
            test_accuracy = metrics.get("test_accuracy", 0.0)
            weighted_precision = metrics.get("test_precision_weighted", 0.0)
            
            # Get hyperparameters
            hyperparameters = log_data.get("best_hyperparameters", {})
            
            # Try to count parameters
            num_params = None
            if os.path.exists(model_path):
                num_params = count_ml_parameters(model_path, model_name)
            
            # Fallback to estimate
            if num_params is None:
                num_params = KNOWN_PARAMETER_ESTIMATES.get(model_name, 500_000)
                print(f"Using parameter estimate for {model_name}: {format_parameters(num_params)}")
            
            results.append({
                "model_name": model_name,
                "model_type": "ML",
                "test_accuracy": test_accuracy,
                "weighted_precision": weighted_precision,
                "num_parameters": num_params,
                "parameters_formatted": format_parameters(num_params),
                "hyperparameters": json.dumps(hyperparameters),
            })
        
        except Exception as e:
            print(f"Error processing {model_dir_name}: {e}")
            continue
    
    return results

def collect_dl_results() -> List[Dict]:
    """Collect results from all DL models."""
    results = []
    
    # Find all DL model directories
    dl_dirs = glob.glob(os.path.join(DL_METHODS_DIR, "*"))
    
    for dl_dir in dl_dirs:
        if not os.path.isdir(dl_dir):
            continue
        
        # Skip __pycache__ and other non-model directories
        if os.path.basename(dl_dir).startswith("__") or os.path.basename(dl_dir).endswith(".py"):
            continue
        
        model_dir_name = os.path.basename(dl_dir)
        log_path = os.path.join(dl_dir, "logs", "evaluation_log.json")
        model_path = os.path.join(dl_dir, "model")
        
        if not os.path.exists(log_path):
            print(f"Warning: No evaluation log found for {model_dir_name}")
            continue
        
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            model_name = log_data.get("model_name", model_dir_name)
            
            # Extract metrics
            metrics = log_data.get("metrics", {})
            test_accuracy = metrics.get("test_accuracy", 0.0)
            weighted_precision = metrics.get("test_precision_weighted", 0.0)
            
            # Get hyperparameters
            hyperparameters = log_data.get("best_hyperparameters", {})
            
            # Try to count parameters
            num_params = None
            if os.path.exists(model_path):
                num_params = count_dl_parameters(model_path, model_name)
            
            # Fallback to estimate
            if num_params is None:
                num_params = KNOWN_PARAMETER_ESTIMATES.get(model_name, 100_000_000)
                print(f"Using parameter estimate for {model_name}: {format_parameters(num_params)}")
            
            results.append({
                "model_name": model_name,
                "model_type": "DL",
                "test_accuracy": test_accuracy,
                "weighted_precision": weighted_precision,
                "num_parameters": num_params,
                "parameters_formatted": format_parameters(num_params),
                "hyperparameters": json.dumps(hyperparameters),
            })
        
        except Exception as e:
            print(f"Error processing {model_dir_name}: {e}")
            continue
    
    return results

# =========================
# Main Execution
# =========================

def main():
    """Main execution function."""
    print("=" * 70)
    print("Model Results Aggregation")
    print("=" * 70)
    print()
    
    # Collect results
    print("Collecting ML model results...")
    ml_results = collect_ml_results()
    print(f"Found {len(ml_results)} ML models")
    print()
    
    print("Collecting DL model results...")
    dl_results = collect_dl_results()
    print(f"Found {len(dl_results)} DL models")
    print()
    
    # Combine all results
    all_results = ml_results + dl_results
    
    if len(all_results) == 0:
        print("No results found. Exiting.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by test accuracy (descending)
    df = df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    
    # Format accuracy and precision as percentages
    df["test_accuracy_pct"] = (df["test_accuracy"] * 100).round(2)
    df["weighted_precision_pct"] = (df["weighted_precision"] * 100).round(2)
    
    # Reorder columns for CSV output
    output_columns = [
        "model_name",
        "model_type",
        "test_accuracy_pct",
        "weighted_precision_pct",
        "num_parameters",
        "parameters_formatted",
        "hyperparameters",
    ]
    
    # Create output DataFrame with renamed columns for clarity
    output_df = df[output_columns].copy()
    output_df.columns = [
        "Model Name",
        "Model Type",
        "Test Accuracy (%)",
        "Weighted Precision (%)",
        "Num Parameters",
        "Parameters (Formatted)",
        "Hyperparameters",
    ]
    
    # Save to CSV
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to: {OUTPUT_CSV}")
    print()
    
    # Print summary statistics
    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print()
    
    print(f"Total Models: {len(all_results)}")
    print(f"  ML Models: {len(ml_results)}")
    print(f"  DL Models: {len(dl_results)}")
    print()
    
    if len(all_results) > 0:
        best_model = df.iloc[0]
        print(f"Best Model: {best_model['model_name']} ({best_model['model_type']})")
        print(f"  Test Accuracy: {best_model['test_accuracy_pct']:.2f}%")
        print(f"  Weighted Precision: {best_model['weighted_precision_pct']:.2f}%")
        print(f"  Parameters: {best_model['parameters_formatted']}")
        print()
        
        avg_accuracy = df["test_accuracy"].mean() * 100
        avg_precision = df["weighted_precision"].mean() * 100
        print(f"Average Test Accuracy: {avg_accuracy:.2f}%")
        print(f"Average Weighted Precision: {avg_precision:.2f}%")
        print()
        
        # ML vs DL comparison
        ml_avg = df[df["model_type"] == "ML"]["test_accuracy"].mean() * 100
        dl_avg = df[df["model_type"] == "DL"]["test_accuracy"].mean() * 100
        print(f"ML Models Average Accuracy: {ml_avg:.2f}%")
        print(f"DL Models Average Accuracy: {dl_avg:.2f}%")
        print()
    
    # Print formatted table
    print("=" * 70)
    print("All Models Results")
    print("=" * 70)
    print()
    
    # Display table with key columns
    display_columns = [
        "Model Name",
        "Model Type",
        "Test Accuracy (%)",
        "Weighted Precision (%)",
        "Parameters (Formatted)",
    ]
    
    print(output_df[display_columns].to_string(index=False))
    print()
    print(f"Full results with hyperparameters saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

