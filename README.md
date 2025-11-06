# MIE1517-Fall2025-Group11

Phishing Email Detection using Machine Learning and Deep Learning

This repository contains a comprehensive pipeline for detecting phishing emails using both traditional machine learning models and deep learning (BERT) approaches. The project is developed for MIE1517 Fall 2025 at the University of Toronto.

**Note**: This repository is set up for local machine development. For compute cluster usage, configure the environment according to your cluster's module system and requirements.

## Dataset

The dataset used for this project can be downloaded from:
**[Download Dataset](https://drive.google.com/file/d/1TS3roaFfIqD8udQyhxWzYVH1QKHT-clL/view?usp=drive_link)**

After downloading, extract the dataset to the `Dataset/` directory in the project root. The data will be processed by the pipeline scripts.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Script Documentation](#script-documentation)
- [Dependencies](#dependencies)
- [Usage Instructions](#usage-instructions)
- [Output Files](#output-files)

## Project Overview

This project implements a two-pronged approach to phishing email detection:

1. **Machine Learning Pipeline**: Uses traditional ML models (Logistic Regression, Linear SVC, SGD) with TF-IDF features and metadata engineering
2. **Deep Learning Pipeline**: Uses BERT (Bidirectional Encoder Representations from Transformers) for sequence classification

Both pipelines are trained on the same dataset but with different preprocessing strategies optimized for each approach.

## Project Structure

```
MIE1517-Fall2025-Group11/
├── 01_data_aggregation.py          # Data aggregation and initial preprocessing
├── 02_ml_preprocessing_eda.py      # ML-specific preprocessing and EDA
├── 03_dl_preprocessing_eda.py      # DL-specific preprocessing and EDA
├── 04_ml_pipeline.py               # ML model training and evaluation
├── 05_dl_pipeline.py               # DL model training and evaluation
├── 06_inference_pipeline.py        # Inference and prediction utilities
├── Dataset/                         # Raw data directory
├── cleaned_data/                    # Processed datasets
├── models/                          # Trained models and artifacts
├── eda_outputs/                     # Exploratory data analysis visualizations
└── README.md                        # This file
```

## Pipeline Overview

The complete pipeline consists of 6 sequential scripts:

```
01_data_aggregation.py
    ↓
02_ml_preprocessing_eda.py  ──→ 04_ml_pipeline.py
    ↓
03_dl_preprocessing_eda.py  ──→ 05_dl_pipeline.py
    ↓
06_inference_pipeline.py (uses models from 04 & 05)
```

### Data Flow

1. **01_data_aggregation.py**: Combines raw CSV files → creates master dataset
2. **02_ml_preprocessing_eda.py**: Processes master dataset for ML → creates ML dataset
3. **03_dl_preprocessing_eda.py**: Processes master dataset for DL → creates DL dataset
4. **04_ml_pipeline.py**: Trains ML models on ML dataset → saves ML model
5. **05_dl_pipeline.py**: Trains DL model on DL dataset → saves DL model
6. **06_inference_pipeline.py**: Loads both models → provides inference interface

## Script Documentation

### 01_data_aggregation.py

**Purpose**: Initial data aggregation and preliminary preprocessing

**What it does**:
- Combines CSV files from multiple directories
- Consolidates text and label columns from different sources
- Standardizes labels (maps to binary: 0=Not Phishing, 1=Phishing)
- Applies initial text cleaning (HTML removal, URL/email replacement)
- Deduplicates the dataset

**Input**: Raw CSV files in `Dataset/raw - DO NOT OVERWRITE/`

**Output**: `cleaned_data/master_email_dataset_final.csv`

**Usage**:
```bash
python 01_data_aggregation.py
```

**Key Features**:
- Multiprocessing support for parallel text cleaning
- Handles multiple column name variations across source files
- Preserves provenance information (source_file, source_name, record_id)

---

### 02_ml_preprocessing_eda.py

**Purpose**: ML-specific preprocessing and exploratory data analysis

**What it does**:
- Aggressive text cleaning (lowercase, removes digits/punctuation) optimized for ML models
- Deduplication
- Length filtering (5-2000 tokens)
- Generates EDA visualizations:
  - Label distribution
  - Text length distributions
  - N-gram analysis
  - Word clouds

**Input**: `cleaned_data/master_email_dataset_final.csv`

**Output**: 
- `cleaned_data/ml_dataset_final.csv`
- `eda_outputs/ml_*.png` (EDA visualizations)

**Usage**:
```bash
python 02_ml_preprocessing_eda.py
```

**Key Features**:
- Aggressive cleaning suitable for TF-IDF features
- Comprehensive EDA visualizations
- Configurable token length filtering

---

### 03_dl_preprocessing_eda.py

**Purpose**: DL-specific preprocessing and exploratory data analysis

**What it does**:
- Gentle text cleaning (preserves case and punctuation) for transformer models
- Deduplication
- Length filtering (5-2000 tokens)
- Generates DL-specific EDA visualizations

**Input**: `cleaned_data/master_email_dataset_final.csv`

**Output**: 
- `cleaned_data/dl_dataset_final.csv`
- `eda_outputs/dl_*.png` (EDA visualizations)

**Usage**:
```bash
python 03_dl_preprocessing_eda.py
```

**Key Features**:
- Preserves case and punctuation (important for BERT)
- Separate preprocessing strategy from ML pipeline
- Similar EDA outputs as ML preprocessing

---

### 04_ml_pipeline.py

**Purpose**: Train and evaluate machine learning models

**What it does**:
1. Loads preprocessed ML dataset
2. Engineers metadata features (word count, punctuation count, uppercase count)
3. Creates train/test split (80/20)
4. Trains multiple models:
   - Logistic Regression
   - Linear SVC (SVM)
   - SGD Classifier
5. Performs cross-validation
6. Hyperparameter tuning with grid search
7. Evaluates models with comprehensive metrics
8. Saves best model for inference

**Input**: `cleaned_data/ml_dataset_final.csv`

**Output**:
- `models/ml_best_model.pkl` (best trained model)
- `models/ml_learning_curve.png`
- `models/ml_evaluation_log.json`
- `models/ml_evaluation_report.txt`
- `models/ml_confusion_matrix.png`

**Usage**:
```bash
python 04_ml_pipeline.py
```

**Key Features**:
- Multiple model comparison
- Cross-validation for robust evaluation
- Grid search for hyperparameter optimization
- Comprehensive evaluation metrics and visualizations
- Feature engineering with TF-IDF and metadata

**Model Details**:
- Text features: TF-IDF with unigrams and bigrams (max 40k features)
- Numeric features: Word count, punctuation count, uppercase count
- Preprocessing: StandardScaler for numeric features

---

### 05_dl_pipeline.py

**Purpose**: Train and evaluate BERT-based deep learning model

**What it does**:
1. Loads preprocessed DL dataset
2. Performs data leakage checks (duplicate detection between train/test)
3. Creates PyTorch datasets and data loaders
4. Sets up BERT model for sequence classification
5. Trains with:
   - Weighted random sampling for class balance
   - Class weights in loss function
   - Mixed precision training (FP16)
   - Learning rate scheduling
6. Evaluates on test set after each epoch
7. Saves best model (highest test accuracy) and final model
8. Saves training curves, evaluation logs, and visualizations

**Input**: `cleaned_data/dl_dataset_final.csv`

**Output**:
- `models/dl_model_best/` (best model based on test accuracy)
- `models/dl_model/` (final model after all epochs)
- `models/dl_tokenizer/` (BERT tokenizer)
- `models/dl_label_encoder.pkl` (label encoder)
- `models/dl_training_history.json` (training metrics per epoch)
- `models/dl_training_curve.png` (loss and accuracy plots)
- `models/dl_evaluation_log.json` (comprehensive metrics)
- `models/dl_evaluation_report.txt` (human-readable report)
- `models/dl_confusion_matrix.png` (confusion matrix visualization)

**Usage**:
```bash
python 05_dl_pipeline.py
```

**Note**: GPU recommended for training (4+ hours on CPU). Ensure CUDA is available if using GPU.

**Key Features**:
- BERT-based sequence classification
- Automatic mixed precision training
- Class imbalance handling
- Data leakage prevention
- Comprehensive logging and visualization
- Early stopping based on test accuracy

**Model Details**:
- Base model: BERT (bert-base-cased)
- Max sequence length: 512 tokens
- Batch size: Configurable (default: 16)
- Learning rate: 2e-5 with warmup
- Optimizer: AdamW
- Loss: CrossEntropyLoss with class weights

---

### 06_inference_pipeline.py

**Purpose**: Load trained models and make predictions

**What it does**:
1. Loads ML model (sklearn pipeline)
2. Loads DL model (BERT), tokenizer, and label encoder
3. Provides functions for:
   - Single predictions with ML model
   - Single predictions with DL model
   - Batch predictions with either model
   - Ensemble predictions combining both models

**Input**: Trained models from `models/` directory

**Output**: Predictions (labels, confidence scores, class indices)

**Usage**:
```bash
# Run example
python 06_inference_pipeline.py

# Or import in Python
from 06_inference_pipeline import predict_ml, predict_dl, predict_ensemble

# ML prediction
label, confidence, class_int = predict_ml("Your email text here...")

# DL prediction
label, confidence, class_int = predict_dl("Your email text here...")

# Ensemble prediction
label, confidence, ml_pred, dl_pred = predict_ensemble("Your email text here...")
```

**Key Features**:
- Handles models without `predict_proba` (e.g., LinearSVC)
- Converts decision scores to probabilities for SVM models
- Supports both single and batch predictions
- Ensemble method combining ML and DL predictions
- Automatic model loading if paths are not specified

**Functions**:
- `predict_ml(text, model=None)`: ML model prediction
- `predict_ml_batch(texts, model=None)`: Batch ML predictions
- `predict_dl(text, model=None, tokenizer=None, label_encoder=None)`: DL model prediction
- `predict_dl_batch(texts, ...)`: Batch DL predictions
- `predict_ensemble(text, ...)`: Weighted ensemble prediction

## Dependencies

### Installation

Install all dependencies using the requirements file:

```bash
# Create virtual environment (recommended)
python -m venv ENV
source ENV/bin/activate  # On Windows: ENV\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for preprocessing)
python -c "import nltk; nltk.download('stopwords')"
```

### Python Packages

Core packages (see `requirements.txt` for versions):
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning models and utilities
- `torch` - PyTorch for deep learning
- `transformers` - Hugging Face transformers (BERT)
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `nltk` - Natural language processing
- `beautifulsoup4` - HTML parsing
- `wordcloud` - Word cloud generation
- `tqdm` - Progress bars
- `scipy` - Scientific computing (optional, has fallback)

### Environment Setup

**For GPU support (optional but recommended for DL pipeline on local machines):**

Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/). For example:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Note**: This setup is for local machines. For compute clusters, use the cluster's module system and environment management tools.

## Usage Instructions

### Complete Pipeline Execution

Run scripts in order:

```bash
# Step 1: Aggregate raw data
python 01_data_aggregation.py

# Step 2: Preprocess for ML
python 02_ml_preprocessing_eda.py

# Step 3: Preprocess for DL
python 03_dl_preprocessing_eda.py

# Step 4: Train ML models
python 04_ml_pipeline.py

# Step 5: Train DL model (GPU recommended)
python 05_dl_pipeline.py

# Step 6: Test inference
python 06_inference_pipeline.py
```

### Quick Start (Inference Only)

If models are already trained:

```bash
python 06_inference_pipeline.py
```

This will run example predictions using both ML and DL models.

## Output Files

### Data Files
- `cleaned_data/master_email_dataset_final.csv` - Aggregated and cleaned master dataset
- `cleaned_data/ml_dataset_final.csv` - ML-ready dataset
- `cleaned_data/dl_dataset_final.csv` - DL-ready dataset

### Model Files
- `models/ml_best_model.pkl` - Best ML model (pickled sklearn pipeline)
- `models/dl_model_best/` - Best DL model (BERT checkpoint)
- `models/dl_model/` - Final DL model (after all epochs)
- `models/dl_tokenizer/` - BERT tokenizer
- `models/dl_label_encoder.pkl` - Label encoder for DL model

### Evaluation Files
- `models/ml_evaluation_log.json` - ML evaluation metrics
- `models/ml_evaluation_report.txt` - ML classification report
- `models/ml_confusion_matrix.png` - ML confusion matrix
- `models/ml_learning_curve.png` - ML learning curve
- `models/dl_evaluation_log.json` - DL evaluation metrics
- `models/dl_evaluation_report.txt` - DL classification report
- `models/dl_confusion_matrix.png` - DL confusion matrix
- `models/dl_training_curve.png` - DL training curves
- `models/dl_training_history.json` - DL per-epoch metrics

### EDA Files
- `eda_outputs/ml_*.png` - ML dataset EDA visualizations
- `eda_outputs/dl_*.png` - DL dataset EDA visualizations

## Notes

- The ML and DL pipelines use different preprocessing strategies:
  - **ML**: Aggressive cleaning (lowercase, removes punctuation) for TF-IDF
  - **DL**: Gentle cleaning (preserves case/punctuation) for BERT
- Both pipelines are trained on the same source data but with different preprocessing
- The inference pipeline handles models without `predict_proba` (e.g., LinearSVC) by converting decision scores to probabilities
- For best DL performance, use a GPU for training (4+ hours on CPU, much faster on GPU)
- All scripts include progress bars and detailed logging

## License

See LICENSE file for details.

## Contact

Group 11 - MIE1517 Fall 2025, University of Toronto
