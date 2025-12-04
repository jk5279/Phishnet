# MIE1517-Fall2025-Group11

Phishing Email Detection using Machine Learning and Deep Learning

This repository contains a comprehensive pipeline for detecting phishing emails using multiple machine learning and deep learning approaches, including traditional ML models (Logistic Regression, SVM, Naive Bayes) and transformer models (BERT, RoBERTa, DistilBERT). The project includes model interpretation tools, an interactive web demo, and a modular architecture for easy extension. Developed for MIE1517 Fall 2025 at the University of Toronto.

![Phishnet Project Image](Phishnet%20Project%20Image.jpg)

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
- [Model Interpretation](#model-interpretation)
- [Model Comparison](#model-comparison)
- [Modular Architecture](#modular-architecture)
- [Notes](#notes)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project implements a comprehensive approach to phishing email detection with multiple models:

1. **Machine Learning Pipeline**: Uses traditional ML models (Logistic Regression, Linear SVC, SGD Classifier, Naive Bayes) with TF-IDF features and metadata engineering
2. **Deep Learning Pipeline**: Uses transformer models (BERT, RoBERTa, DistilBERT) for sequence classification
3. **Model Interpretation**: Attention visualization and analysis tools for understanding model decisions
4. **Interactive Demo**: Streamlit web application for live phishing detection

All pipelines are trained on the same dataset but with different preprocessing strategies optimized for each approach. The project uses a modular architecture that allows easy addition of new models.

## Project Structure

```
MIE1517-Fall2025-Group11/
├── 01_data_aggregation.py          # Data aggregation and initial preprocessing
├── 02_ml_preprocessing_eda.py      # ML-specific preprocessing and EDA
├── 03_dl_preprocessing_eda.py      # DL-specific preprocessing and EDA
├── 04_ml_pipeline.py               # ML model training and evaluation
├── 05_dl_pipeline.py               # DL model training and evaluation (legacy)
├── 06_inference_pipeline.py        # Inference and prediction utilities (legacy)
├── 07_phishing_categories_retrieval.py  # Scrape Berkeley phishing examples archive
├── 08_process_hunter_biden_email_dataset.py  # Process Hunter Biden email dataset
├── 09_inference.py                 # RoBERTa inference on demonstration dataset
├── 10_model_interpretation.py      # Attention visualization and model interpretation
├── 11_aggregate_results.py          # Aggregate results from all models
├── 12_test_attention_visualization.py  # Attention visualization for test samples
├── check_data_completeness.py      # Data quality checking utility
├── streamlit_app.py                # Interactive web demo application
├── datasets/                        # Raw data directory
├── cleaned_data/                    # Processed datasets
│   ├── ML/                          # ML-specific processed data
│   └── DL/                          # DL-specific processed data
├── ml_methods/                      # Modular ML model implementations
│   ├── logistic_regression/         # Logistic Regression model
│   ├── linear_svc/                  # Linear SVC model
│   ├── sgd_classifier/              # SGD Classifier model
│   └── naive_bayes/                 # Naive Bayes model
├── dl_methods/                      # Modular DL model implementations
│   ├── base_pipeline.py            # Base class for DL pipelines
│   ├── bert_pipeline.py            # BERT pipeline
│   ├── roberta_pipeline.py         # RoBERTa pipeline
│   ├── distilbert_pipeline.py      # DistilBERT pipeline
│   ├── utils.py                     # Shared utilities
│   ├── bert/                        # BERT model directory
│   ├── roberta/                     # RoBERTa model directory
│   └── distilbert/                  # DistilBERT model directory
├── models/                          # Legacy trained models (deprecated)
├── eda_outputs/                     # Exploratory data analysis visualizations
└── README.md                        # This file
```

## Pipeline Overview

The complete pipeline consists of 12 main scripts organized into data processing, model training, inference, and analysis phases:

```
Data Processing Phase:
01_data_aggregation.py
    ↓
02_ml_preprocessing_eda.py  ──→ 04_ml_pipeline.py (trains multiple ML models)
    ↓
03_dl_preprocessing_eda.py  ──→ dl_methods/*_pipeline.py (trains multiple DL models)

Inference & Analysis Phase:
09_inference.py (RoBERTa inference on demo dataset)
    ↓
10_model_interpretation.py (attention visualization)
    ↓
12_test_attention_visualization.py (test set attention analysis)
    ↓
11_aggregate_results.py (aggregate all model results)
```

### Data Flow

1. **01_data_aggregation.py**: Combines raw CSV files → creates processed datasets and train/val/test splits
2. **02_ml_preprocessing_eda.py**: Processes master dataset for ML → creates ML dataset in `cleaned_data/ML/`
3. **03_dl_preprocessing_eda.py**: Processes master dataset for DL → creates DL dataset in `cleaned_data/DL/`
4. **04_ml_pipeline.py**: Trains multiple ML models (Logistic Regression, Linear SVC, SGD, Naive Bayes) → saves to `ml_methods/{model_name}/`
5. **DL Pipelines** (`dl_methods/*_pipeline.py`): Train transformer models (BERT, RoBERTa, DistilBERT) → saves to `dl_methods/{model_name}/`
6. **09_inference.py**: Runs RoBERTa inference on demonstration dataset
7. **10_model_interpretation.py**: Generates attention visualizations for model interpretation
8. **12_test_attention_visualization.py**: Creates attention visualizations for test samples
9. **11_aggregate_results.py**: Aggregates results from all models into summary CSV

### Modular Architecture

The project uses a modular architecture for deep learning models:

- **BaseDLPipeline** (`dl_methods/base_pipeline.py`): Abstract base class that handles common training logic, evaluation, and model saving
- **Model-specific pipelines**: Each transformer model (BERT, RoBERTa, DistilBERT) has its own pipeline class that inherits from `BaseDLPipeline` and implements model-specific initialization
- **Shared utilities** (`dl_methods/utils.py`): Common functions for data loading, evaluation, and visualization

This architecture makes it easy to add new transformer models by simply creating a new pipeline class.

## Script Documentation

### 01_data_aggregation.py

**Purpose**: Initial data aggregation and preliminary preprocessing

**What it does**:
- Stage 1: Aggregates CSV files from `datasets/raw - DO NOT OVERWRITE/` (excluding `test_datasets/`)
- Stage 2: Aggregates CSV files from `test_datasets/` directory
- For both stages: consolidates columns, standardizes labels, cleans text, deduplicates
- Samples from dataset1 to reach target size (default: 100,000 rows), combines with dataset2
- Splits combined dataset into train (70%), validation (15%), and test (15%) with stratified sampling

**Input**: 
- Raw CSV files in `datasets/raw - DO NOT OVERWRITE/` (excluding `test_datasets/`)
- Test datasets in `datasets/raw - DO NOT OVERWRITE/test_datasets/`

**Output**: 
- `cleaned_data/dataset1_processed.csv` - Processed training source data
- `cleaned_data/dataset2_processed.csv` - Processed test datasets
- `cleaned_data/dataset3_combined.csv` - Combined sampled dataset
- `cleaned_data/train/train_split.csv` - Training set (70%)
- `cleaned_data/validation/validation_split.csv` - Validation set (15%)
- `cleaned_data/test/test_split.csv` - Test set (15%)

**Usage**:
```bash
python 01_data_aggregation.py
```

**Key Features**:
- Two-stage aggregation (separates training data from test datasets)
- Intelligent sampling (samples from dataset1 to reach target size)
- Stratified train/validation/test split
- Minimum text length filtering (50 characters)
- Support for lowercase "text" and "label" columns (test_datasets format)
- All configuration values extracted to named constants
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

**Input**: `cleaned_data/dataset3_combined.csv` (from 01_data_aggregation.py)

**Output**: 
- `cleaned_data/ml_cleaning/ml_dataset_final.csv`
- `eda_outputs/ml_cleaning/*.png` (EDA visualizations)

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

**Input**: `cleaned_data/dataset3_combined.csv` (from 01_data_aggregation.py)

**Output**: 
- `cleaned_data/dl_cleaning/dl_dataset_final.csv`
- `eda_outputs/dl_cleaning/dl_*.png` (EDA visualizations)

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

**Purpose**: Train and evaluate multiple machine learning models

**What it does**:
1. Loads preprocessed ML dataset from `cleaned_data/ML/` splits
2. Engineers metadata features (word count, punctuation count, uppercase count)
3. Trains multiple models:
   - Logistic Regression
   - Linear SVC (SVM)
   - SGD Classifier
   - Naive Bayes
4. Performs cross-validation for each model
5. Hyperparameter tuning with grid search
6. Evaluates models with comprehensive metrics
7. Saves each model to its own directory in `ml_methods/`

**Input**: 
- `cleaned_data/ML/train/train_split.csv`
- `cleaned_data/ML/validation/validation_split.csv`
- `cleaned_data/ML/test/test_split.csv`

**Output**:
- `ml_methods/{model_name}/model.pkl` - Trained model for each method
- `ml_methods/{model_name}/logs/evaluation_log.json` - Evaluation metrics
- `ml_methods/{model_name}/logs/evaluation_report.txt` - Classification report
- `ml_methods/{model_name}/logs/hyperparameters.json` - Best hyperparameters

**Usage**:
```bash
python 04_ml_pipeline.py
```

**Key Features**:
- Multiple model comparison (4 different ML algorithms)
- Cross-validation for robust evaluation
- Grid search for hyperparameter optimization
- Comprehensive evaluation metrics and visualizations
- Feature engineering with TF-IDF and metadata
- Modular output structure (one directory per model)

**Model Details**:
- Text features: TF-IDF with unigrams and bigrams (max 40k features)
- Numeric features: Word count, punctuation count, uppercase count
- Preprocessing: StandardScaler for numeric features

---

### 05_dl_pipeline.py

**Purpose**: Legacy script for training BERT model (deprecated - use modular pipelines instead)

**Note**: This script is kept for backward compatibility. New projects should use the modular DL pipelines in `dl_methods/` directory.

**Alternative**: Use `dl_methods/bert_pipeline.py`, `dl_methods/roberta_pipeline.py`, or `dl_methods/distilbert_pipeline.py` for training transformer models.

---

### Deep Learning Modular Pipelines

The project includes modular deep learning pipelines for multiple transformer models:

#### dl_methods/base_pipeline.py

**Purpose**: Abstract base class for all DL pipelines

**Key Features**:
- Common training loop with mixed precision support
- Automatic model saving (best and final checkpoints)
- Data leakage detection
- Class weight calculation for imbalanced datasets
- Comprehensive evaluation and logging
- Training history tracking

#### dl_methods/roberta_pipeline.py

**Purpose**: Train and evaluate RoBERTa model (recommended - best performance)

**What it does**:
1. Loads preprocessed DL dataset from `cleaned_data/DL/` splits
2. Performs data leakage checks
3. Creates PyTorch datasets and data loaders
4. Sets up RoBERTa model for sequence classification
5. Trains with mixed precision, class weights, and learning rate scheduling
6. Saves best model (highest validation accuracy) and final model
7. Generates comprehensive evaluation reports

**Input**: 
- `cleaned_data/DL/train/train_split.csv`
- `cleaned_data/DL/validation/validation_split.csv`
- `cleaned_data/DL/test/test_split.csv`

**Output**:
- `dl_methods/roberta/model/best/` - Best model checkpoint
- `dl_methods/roberta/model/final/` - Final model after all epochs
- `dl_methods/roberta/model/tokenizer/` - RoBERTa tokenizer
- `dl_methods/roberta/model/label_encoder.pkl` - Label encoder
- `dl_methods/roberta/logs/training_history.json` - Per-epoch metrics
- `dl_methods/roberta/logs/evaluation_log.json` - Comprehensive metrics
- `dl_methods/roberta/logs/evaluation_report.txt` - Classification report
- `dl_methods/roberta/logs/hyperparameters.json` - Training configuration

**Usage**:
```bash
python -m dl_methods.roberta_pipeline
# Or
cd dl_methods && python roberta_pipeline.py
```

**Model Details**:
- Base model: RoBERTa (roberta-base)
- Max sequence length: 128 tokens (configurable)
- Batch size: 16 (configurable)
- Learning rate: 2e-5 with warmup
- Optimizer: AdamW
- Loss: CrossEntropyLoss with class weights

#### dl_methods/bert_pipeline.py

**Purpose**: Train and evaluate BERT model

Similar structure to RoBERTa pipeline but uses BERT-base-cased. Outputs to `dl_methods/bert/`.

#### dl_methods/distilbert_pipeline.py

**Purpose**: Train and evaluate DistilBERT model (faster, smaller)

Similar structure to RoBERTa pipeline but uses DistilBERT. Outputs to `dl_methods/distilbert/`.

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

---

### 07_phishing_categories_retrieval.py

**Purpose**: Scrape Berkeley phishing examples archive to retrieve phishing email categories and descriptions

**What it does**:
1. Scrapes the UC Berkeley Security website's phishing examples archive
2. Retrieves phishing email categories, titles, and descriptions
3. Follows links to individual phishing examples to extract detailed information
4. Extracts "What makes this a phishing message?" sections for each category
5. Saves all retrieved data to a CSV file for further processing

**Input**: UC Berkeley Security website (https://security.berkeley.edu/education-awareness/phishing/phishing-examples-archive)

**Output**: `phishing_examples_berkley.csv` - CSV file containing:
- Title of each phishing category
- Link to the detailed page
- Description of the phishing example
- "What makes this a phishing message?" content

**Usage**:
```bash
python 07_phishing_categories_retrieval.py
```

**Key Features**:
- Web scraping with BeautifulSoup
- Handles dynamic content and nested HTML structures
- Includes rate limiting (sleep between requests) to be respectful to the server
- Extracts structured information from multiple pages
- Sanity checks to ensure relevant content is retrieved

**Note**: This script is used to gather context for generating synthetic phishing emails. The retrieved categories and descriptions can be used as prompts for LLM-based email generation.

---

### 08_process_hunter_biden_email_dataset.py

**Purpose**: Process the Hunter Biden email dataset to extract legitimate emails for training

**What it does**:
1. Reads the Hunter Biden email dataset (JSON format) from Kaggle
2. Extracts individual emails from the dataset
3. Filters emails by length (removes overly long emails > 500 characters)
4. Labels all emails as "Not Phishing" (legitimate communications)
5. Saves extracted emails to CSV format for integration with other datasets

**Input**: Hunter Biden email dataset (JSON file)
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/anuranroy/hunter-biden-mails)
- **Note**: The dataset is large (~3GB) and not stored in this repository

**Output**: `datasets/raw - DO NOT OVERWRITE/hunter_biden_emails/hunter_biden_1000_emails.csv`
- CSV file with `text` and `label` columns
- Default: 1000 emails extracted (configurable)

**Usage**:
```bash
# Update the dataset_path variable in the script to point to your downloaded dataset
python 08_process_hunter_biden_email_dataset.py
```

**Key Features**:
- Uses `ijson` for efficient streaming JSON parsing (handles large files)
- Extracts individual emails from email chains by finding email boundaries
- Filters by email length to maintain quality
- Configurable number of emails to extract
- Automatically creates output directory if it doesn't exist

**Configuration**:
- `num_emails`: Number of emails to extract (default: 1000)
- `dataset_path`: Path to the downloaded JSON dataset file
- `csv_output_path`: Output directory for the processed CSV file

**Note**: This script adds legitimate email examples to the training dataset, helping balance the dataset with non-phishing emails. The processed CSV can be merged with other datasets using `01_data_aggregation.py`.

---

### 09_inference.py

**Purpose**: Run RoBERTa inference on demonstration dataset

**What it does**:
1. Loads the best trained RoBERTa model, tokenizer, and label encoder
2. Loads demonstration dataset from `datasets/Demonstration dataset.csv`
3. Combines Title and Text columns for inference
4. Runs batch inference with confidence scores
5. Saves results as JSON with predictions and probabilities

**Input**: 
- `datasets/Demonstration dataset.csv` (must contain 'Title' and 'Text' columns)
- RoBERTa model from `dl_methods/roberta/model/best/`

**Output**: 
- `dl_methods/roberta/logs/demonstration_dataset_predictions.json` - Complete inference results with:
  - Individual predictions for each sample
  - Confidence scores and class probabilities
  - Summary statistics (accuracy, prediction distribution)

**Usage**:
```bash
python 09_inference.py
```

**Key Features**:
- Batch inference for efficient processing
- Confidence scores and probability distributions
- Summary statistics and accuracy calculation
- JSON output for easy integration with other tools

---

### 10_model_interpretation.py

**Purpose**: Generate attention visualizations for model interpretation

**What it does**:
1. Loads RoBERTa model and predictions from `09_inference.py`
2. Extracts attention weights from the CLS token
3. Reconstructs subword tokens into whole words
4. Generates attention heatmaps showing which words the model focuses on
5. Creates bar charts of top attention words
6. Separates correct and incorrect predictions for analysis

**Input**: 
- `dl_methods/roberta/logs/demonstration_dataset_predictions.json` (from `09_inference.py`)
- RoBERTa model from `dl_methods/roberta/model/best/`

**Output**: 
- `dl_methods/roberta/logs/interpretations/attention_correct/` - Visualizations for correct predictions
- `dl_methods/roberta/logs/interpretations/attention_incorrect/` - Visualizations for incorrect predictions
  - `heat_{sample_id}.png` - Attention heatmap visualization
  - `bar_{sample_id}.png` - Top attention words bar chart

**Usage**:
```bash
python 10_model_interpretation.py
```

**Key Features**:
- Attention weight extraction from transformer layers
- Subword token reconstruction (handles RoBERTa's tokenization)
- Color-coded heatmaps showing attention intensity
- Separate analysis for correct vs incorrect predictions
- High-resolution output (600 DPI) for publication quality

---

### 11_aggregate_results.py

**Purpose**: Aggregate results from all ML and DL models into a comprehensive summary

**What it does**:
1. Scans `ml_methods/` and `dl_methods/` directories for all trained models
2. Loads evaluation logs from each model
3. Extracts key metrics (test accuracy, weighted precision)
4. Counts model parameters (actual count for DL models, estimates for ML)
5. Generates a CSV report with all models ranked by performance
6. Prints summary statistics and comparison tables

**Input**: 
- Evaluation logs from `ml_methods/*/logs/evaluation_log.json`
- Evaluation logs from `dl_methods/*/logs/evaluation_log.json`
- Model files for parameter counting

**Output**: 
- `model_results_summary.csv` - Comprehensive CSV report with columns:
  - Model Name
  - Model Type (ML/DL)
  - Test Accuracy (%)
  - Weighted Precision (%)
  - Number of Parameters
  - Hyperparameters

**Usage**:
```bash
python 11_aggregate_results.py
```

**Key Features**:
- Automatic discovery of all trained models
- Parameter counting for both ML and DL models
- Human-readable parameter formatting (K/M/B)
- Sorted by test accuracy for easy comparison
- Summary statistics (averages, best model identification)

---

### 12_test_attention_visualization.py

**Purpose**: Generate attention visualizations for test dataset samples

**What it does**:
1. Loads test dataset from `cleaned_data/DL/test/test_split.csv`
2. Runs inference using RoBERTa model on selected samples
3. Identifies correct and incorrect predictions
4. Selects diverse samples (mix of labels and prediction correctness)
5. Generates attention visualizations for selected samples
6. Saves metadata about selected samples

**Input**: 
- `cleaned_data/DL/test/test_split.csv`
- RoBERTa model from `dl_methods/roberta/model/best/`

**Output**: 
- `dl_methods/roberta/logs/interpretations/test_samples/` - Attention visualizations
  - `test_sample_{id}.png` - Attention heatmap for each sample
  - `metadata.json` - Information about selected samples (true labels, predictions, correctness)

**Usage**:
```bash
python 12_test_attention_visualization.py
```

**Key Features**:
- Automatic sample selection with label diversity
- Inference and visualization in one script
- Metadata tracking for reproducibility
- Visual indicators for prediction correctness

---

### check_data_completeness.py

**Purpose**: Data quality checking utility

**What it does**:
1. Checks all train/validation/test split CSV files
2. Identifies missing labels, source_file, or source_name values
3. Detects extremely long text fields (>100k or >1M characters)
4. Reports data completeness issues that might affect training or CSV viewing

**Input**: 
- `cleaned_data/train/train_split.csv`
- `cleaned_data/validation/validation_split.csv`
- `cleaned_data/test/test_split.csv`

**Output**: Console report with:
- Count of empty labels, source_file, source_name
- List of rows with missing data
- Identification of extremely long text fields
- Overall data completeness summary

**Usage**:
```bash
python check_data_completeness.py
```

**Key Features**:
- Comprehensive data quality checks
- Identifies potential issues before training
- Handles large CSV files efficiently
- Clear reporting of issues with row numbers

---

### streamlit_app.py

**Purpose**: Interactive web application for live phishing email detection

**What it does**:
1. Loads the best RoBERTa model (cached for performance)
2. Provides a web interface for entering email text
3. Runs real-time inference on user input
4. Displays prediction (Safe/Phishing) with confidence score
5. Shows model internals (probabilities, logits) in expandable section
6. Displays model performance metrics in sidebar

**Input**: User-entered email text via web interface

**Output**: Real-time predictions displayed in web browser

**Usage**:
```bash
# Install streamlit if not already installed
pip install streamlit

# Run the app
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

**Key Features**:
- Real-time inference with instant feedback
- User-friendly interface with clear visual indicators
- Model performance metrics displayed in sidebar
- Expandable section showing model internals
- Cached model loading for fast startup

**Note**: Requires Streamlit to be installed (`pip install streamlit`). The app uses the RoBERTa model from `dl_methods/roberta/model/best/`.

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
- `transformers` - Hugging Face transformers (BERT, RoBERTa, DistilBERT)
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical visualizations
- `nltk` - Natural language processing
- `beautifulsoup4` - HTML parsing
- `wordcloud` - Word cloud generation
- `tqdm` - Progress bars
- `scipy` - Scientific computing (optional, has fallback)
- `requests` - HTTP library for web scraping (script 07)
- `ijson` - Streaming JSON parser for large files (script 08)
- `shap` - Model interpretation (SHAP values)

**Additional package for Streamlit app** (not in requirements.txt):
- `streamlit` - Web application framework (install with `pip install streamlit`)

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

# Step 4: Train ML models (trains all 4 models)
python 04_ml_pipeline.py

# Step 5: Train DL models (choose one or all)
# RoBERTa (recommended - best performance)
python -m dl_methods.roberta_pipeline
# Or BERT
python -m dl_methods.bert_pipeline
# Or DistilBERT (faster, smaller)
python -m dl_methods.distilbert_pipeline

# Step 6: Run inference on demonstration dataset
python 09_inference.py

# Step 7: Generate model interpretations
python 10_model_interpretation.py

# Step 8: Generate test set attention visualizations
python 12_test_attention_visualization.py

# Step 9: Aggregate all model results
python 11_aggregate_results.py
```

### Quick Start (Inference Only)

If models are already trained:

```bash
# Run RoBERTa inference on demonstration dataset
python 09_inference.py

# Or use the interactive Streamlit app
streamlit run streamlit_app.py
```

### Running the Streamlit Demo

To launch the interactive web application:

```bash
# Install streamlit if not already installed
pip install streamlit

# Run the app
streamlit run streamlit_app.py
```

The app will automatically open in your default web browser. You can enter email text and get real-time phishing detection predictions.

### Model Interpretation

To understand what the model focuses on when making predictions:

```bash
# First, run inference to generate predictions
python 09_inference.py

# Then generate attention visualizations
python 10_model_interpretation.py

# Or generate visualizations for test set samples
python 12_test_attention_visualization.py
```

Visualizations will be saved in `dl_methods/roberta/logs/interpretations/`.

### Model Comparison

To compare all trained models:

```bash
python 11_aggregate_results.py
```

This generates `model_results_summary.csv` with all models ranked by performance.

## Output Files

### Data Files
- `cleaned_data/dataset1_processed.csv` - Processed training source data
- `cleaned_data/dataset2_processed.csv` - Processed test datasets
- `cleaned_data/dataset3_combined.csv` - Combined sampled dataset
- `cleaned_data/train/train_split.csv` - Training set (70%) - Legacy location
- `cleaned_data/validation/validation_split.csv` - Validation set (15%) - Legacy location
- `cleaned_data/test/test_split.csv` - Test set (15%) - Legacy location
- `cleaned_data/ML/train/train_split.csv` - ML training set (70%)
- `cleaned_data/ML/validation/validation_split.csv` - ML validation set (15%)
- `cleaned_data/ML/test/test_split.csv` - ML test set (15%)
- `cleaned_data/DL/train/train_split.csv` - DL training set (70%)
- `cleaned_data/DL/validation/validation_split.csv` - DL validation set (15%)
- `cleaned_data/DL/test/test_split.csv` - DL test set (15%)
- `cleaned_data/ml_cleaning/ml_dataset_final.csv` - ML-ready dataset (legacy)
- `cleaned_data/dl_cleaning/dl_dataset_final.csv` - DL-ready dataset (legacy)

### ML Model Files (ml_methods/)
Each ML model has its own directory with the following structure:

- `ml_methods/{model_name}/model.pkl` - Trained model (pickled sklearn pipeline)
- `ml_methods/{model_name}/logs/evaluation_log.json` - Evaluation metrics
- `ml_methods/{model_name}/logs/evaluation_report.txt` - Classification report
- `ml_methods/{model_name}/logs/hyperparameters.json` - Best hyperparameters

Available models: `logistic_regression`, `linear_svc`, `sgd_classifier`, `naive_bayes`

### DL Model Files (dl_methods/)
Each DL model has its own directory with the following structure:

- `dl_methods/{model_name}/model/best/` - Best model checkpoint (highest validation accuracy)
- `dl_methods/{model_name}/model/final/` - Final model after all epochs
- `dl_methods/{model_name}/model/tokenizer/` - Model tokenizer
- `dl_methods/{model_name}/model/label_encoder.pkl` - Label encoder
- `dl_methods/{model_name}/logs/training_history.json` - Per-epoch training metrics
- `dl_methods/{model_name}/logs/evaluation_log.json` - Comprehensive evaluation metrics
- `dl_methods/{model_name}/logs/evaluation_report.txt` - Classification report
- `dl_methods/{model_name}/logs/hyperparameters.json` - Training configuration

Available models: `bert`, `roberta`, `distilbert`

### Legacy Model Files (models/)
These are from the old pipeline structure (deprecated):
- `models/ml_best_model.pkl` - Legacy ML model
- `models/dl_model_best/` - Legacy DL model
- `models/dl_model/` - Legacy final DL model
- `models/dl_tokenizer/` - Legacy tokenizer
- `models/dl_label_encoder.pkl` - Legacy label encoder

### Inference and Interpretation Files
- `dl_methods/roberta/logs/demonstration_dataset_predictions.json` - Inference results from `09_inference.py`
- `dl_methods/roberta/logs/interpretations/attention_correct/` - Attention visualizations for correct predictions
- `dl_methods/roberta/logs/interpretations/attention_incorrect/` - Attention visualizations for incorrect predictions
- `dl_methods/roberta/logs/interpretations/test_samples/` - Test set attention visualizations
- `dl_methods/roberta/logs/interpretations/test_samples/metadata.json` - Metadata for test samples

### Aggregation Files
- `model_results_summary.csv` - Comprehensive comparison of all models (from `11_aggregate_results.py`)
- `all_models_inference_results.json` - Inference results comparison (if generated)

### EDA Files
- `eda_outputs/ml_cleaning/*.png` - ML dataset EDA visualizations
- `eda_outputs/dl_cleaning/dl_*.png` - DL dataset EDA visualizations

## Model Interpretation

The project includes comprehensive model interpretation tools to understand what the models focus on when making predictions:

### Attention Visualization

The RoBERTa model uses attention mechanisms to focus on different parts of the input text. The interpretation scripts extract and visualize these attention weights:

- **10_model_interpretation.py**: Generates attention heatmaps for predictions from the demonstration dataset
- **12_test_attention_visualization.py**: Creates attention visualizations for test set samples

The visualizations show:
- **Heatmaps**: Color-coded text where intensity indicates attention weight (darker = more attention)
- **Bar charts**: Top words that receive the most attention from the model
- **Correct vs Incorrect**: Separate analysis for correct and incorrect predictions to understand failure modes

### Understanding Attention Visualizations

- **Yellow/Orange/Red colors**: Higher attention (model focuses more on these words)
- **Pale colors**: Lower attention
- **CLS token attention**: Shows which words the model uses to make its final classification decision

These visualizations help:
- Understand model decision-making process
- Identify important features for phishing detection
- Debug model failures
- Validate model behavior on edge cases

## Model Comparison

The project trains multiple models for comparison:

### Machine Learning Models
1. **Logistic Regression** - Linear classifier with TF-IDF features
2. **Linear SVC** - Support Vector Machine with linear kernel
3. **SGD Classifier** - Stochastic Gradient Descent (SVM-like)
4. **Naive Bayes** - Probabilistic baseline model

### Deep Learning Models
1. **BERT** - Bidirectional Encoder Representations from Transformers (110M parameters)
2. **RoBERTa** - Robustly Optimized BERT (125M parameters) - **Recommended, best performance**
3. **DistilBERT** - Distilled BERT (66M parameters) - Faster, smaller

### Comparing Models

Run `11_aggregate_results.py` to generate a comprehensive comparison:

```bash
python 11_aggregate_results.py
```

This creates `model_results_summary.csv` with:
- Test accuracy for all models
- Weighted precision scores
- Parameter counts
- Hyperparameters used

Models are automatically ranked by test accuracy for easy comparison.

## Modular Architecture

The project uses a modular architecture for deep learning models:

### BaseDLPipeline

All DL models inherit from `BaseDLPipeline` (`dl_methods/base_pipeline.py`), which provides:
- Common training loop with mixed precision support
- Automatic model checkpointing (best and final)
- Data leakage detection
- Class weight calculation
- Comprehensive evaluation and logging
- Training history tracking

### Adding New Models

To add a new transformer model:

1. Create a new pipeline file (e.g., `dl_methods/xlnet_pipeline.py`)
2. Inherit from `BaseDLPipeline`
3. Implement `_create_model()` and `_get_tokenizer()` methods
4. The base class handles all training, evaluation, and saving

Example:
```python
from transformers import XLNetForSequenceClassification, XLNetTokenizer
from .base_pipeline import BaseDLPipeline

class XLNetPipeline(BaseDLPipeline):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="xlnet",
            pretrained_model_name="xlnet-base-cased",
            **kwargs
        )
    
    def _create_model(self, num_classes: int):
        return XLNetForSequenceClassification.from_pretrained(...)
    
    def _get_tokenizer(self):
        return XLNetTokenizer.from_pretrained(...)
```

## Notes

- Configuration values (target dataset size, split ratios, minimum text length, etc.) can be adjusted via constants at the top of the `01_data_aggregation.py` script.
- The ML and DL pipelines use different preprocessing strategies:
  - **ML**: Aggressive cleaning (lowercase, removes punctuation) for TF-IDF
  - **DL**: Gentle cleaning (preserves case/punctuation) for transformers
- Both pipelines are trained on the same source data but with different preprocessing
- The inference pipeline handles models without `predict_proba` (e.g., LinearSVC) by converting decision scores to probabilities
- For best DL performance, use a GPU for training (4+ hours on CPU, much faster on GPU)
- All scripts include progress bars and detailed logging
- **RoBERTa is recommended** as the best-performing model based on evaluation metrics
- Model files are organized in `ml_methods/` and `dl_methods/` directories for better modularity
- Legacy models in `models/` directory are from older pipeline versions

## License

See LICENSE file for details.

## Contact

Group 11 - MIE1517 Fall 2025, University of Toronto
Jongeun Kim (jhnny.kim@mail.utoronto.ca)

