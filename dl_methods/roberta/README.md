# RoBERTa Model Guide

## Overview

This guide provides comprehensive instructions for training and using the **RoBERTa (Robustly Optimized BERT Pretraining Approach)** model for phishing email detection. RoBERTa is the **recommended model** for this project, achieving the best performance among all tested models.

### Model Architecture

- **Base Model**: `roberta-base` (125M parameters)
- **Task**: Binary sequence classification (Safe/Phishing)
- **Architecture**: Transformer-based encoder with classification head
- **Input**: Email text (max 128 tokens)
- **Output**: Binary classification (Great/Safe = 0, Bait/Phishing = 1)

### Key Features

- **Best Performance**: Highest accuracy among all models in the project
- **Mixed Precision Training**: Automatic mixed precision (AMP) for faster training
- **Class Weight Balancing**: Handles imbalanced datasets automatically
- **Comprehensive Logging**: Detailed training history and evaluation metrics
- **Model Checkpointing**: Saves both best and final model checkpoints

### Related Files

- Main project README: [../../README.md](../../README.md)
- Training pipeline: [../roberta_pipeline.py](../roberta_pipeline.py)
- Base pipeline class: [../base_pipeline.py](../base_pipeline.py)
- Inference script: [../../09_inference.py](../../09_inference.py)
- Streamlit app: [../../streamlit_app.py](../../streamlit_app.py)

---

## Prerequisites

### Python Version

- Python 3.8 or higher

### Required Packages

Install all dependencies from the project root:

```bash
pip install -r requirements.txt
```

Key packages for RoBERTa:
- `torch>=2.0.0` - PyTorch deep learning framework
- `transformers>=4.30.0` - Hugging Face transformers library
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.23.0` - Numerical computing
- `scikit-learn>=1.2.0` - Machine learning utilities
- `tqdm>=4.64.0` - Progress bars

### Hardware Requirements

- **GPU Recommended**: Training on GPU is significantly faster (4+ hours on CPU vs. ~30-60 minutes on GPU)
- **Minimum RAM**: 8GB (16GB+ recommended)
- **Disk Space**: ~2GB for model files and checkpoints

### Data Preparation

Before training, ensure you have:
1. Preprocessed datasets in the correct format (see Data Preparation section)
2. Train/validation/test splits ready
3. Data preprocessing completed using `03_dl_preprocessing_eda.py`

---

## Quick Start

### Training

```bash
# From project root
python -m dl_methods.roberta_pipeline

# Or from dl_methods directory
cd dl_methods
python roberta_pipeline.py
```

### Inference

```bash
# Using the inference script
python 09_inference.py

# Or using the Streamlit app
streamlit run streamlit_app.py
```

---

## Data Preparation

### Required Data Format

The model expects CSV files with the following structure:

**Required columns:**
- `text`: Email text content (string)
- `label`: Email label (binary: 0 = Safe/Great, 1 = Phishing/Bait)

**Example CSV structure:**
```csv
text,label
"Your account has been suspended. Click here to verify...",1
"Meeting reminder for tomorrow at 3 PM",0
```

### Expected File Structure

```
cleaned_data/
├── DL/
│   ├── train/
│   │   └── train_split.csv
│   ├── validation/
│   │   └── validation_split.csv
│   └── test/
│       └── test_split.csv
```

### Data Preprocessing

Use the DL preprocessing script to prepare your data:

```bash
# Step 1: Aggregate raw data
python 01_data_aggregation.py

# Step 2: Preprocess for deep learning (preserves case/punctuation)
python 03_dl_preprocessing_eda.py
```

The preprocessing script:
- Cleans text while preserving case and punctuation (important for transformers)
- Filters by length (5-2000 tokens)
- Removes duplicates
- Creates train/validation/test splits (70%/15%/15%)

**Note**: The DL preprocessing preserves case and punctuation, which is important for transformer models like RoBERTa.

---

## Training

### Step-by-Step Training Instructions

1. **Ensure data is prepared** (see Data Preparation section)

2. **Run the training pipeline**:

```bash
# From project root
python -m dl_methods.roberta_pipeline
```

3. **Monitor training progress**:
   - Training loss and accuracy are displayed per epoch
   - Validation metrics are shown after each epoch
   - Best model is automatically saved when validation accuracy improves

4. **Training completes** when all epochs are finished:
   - Best model checkpoint saved
   - Final model checkpoint saved
   - Training history and evaluation reports generated

### Command-Line Usage

The training script can be run directly:

```bash
cd dl_methods
python roberta_pipeline.py
```

### Configuration Options

You can customize training by modifying the pipeline initialization:

```python
from dl_methods.roberta_pipeline import RobertaPipeline

pipeline = RobertaPipeline(
    batch_size=16,        # Batch size (default: 16)
    epochs=4,            # Number of training epochs (default: 4)
    learning_rate=2e-5,  # Learning rate (default: 2e-5)
    max_len=128,          # Maximum sequence length (default: 128)
    random_seed=42        # Random seed for reproducibility (default: 42)
)

pipeline.run(
    train_path="cleaned_data/DL/train/train_split.csv",
    val_path="cleaned_data/DL/validation/validation_split.csv",
    test_path="cleaned_data/DL/test/test_split.csv"
)
```

### Hyperparameter Tuning

**Recommended hyperparameters** (defaults):
- **Batch size**: 16 (reduce to 8 if GPU memory is limited)
- **Epochs**: 4 (increase to 6-8 for better performance, but watch for overfitting)
- **Learning rate**: 2e-5 (standard for transformer fine-tuning)
- **Max sequence length**: 128 (increase to 256 or 512 for longer emails, but slower)

**Tuning tips**:
- Start with default values
- Increase batch size if you have GPU memory available
- Monitor validation loss to detect overfitting
- Use learning rate scheduling (already included)

### Expected Outputs and Logs

During training, you'll see:
- Device information (CPU/GPU)
- Data loading progress
- Per-epoch training and validation metrics
- Best model notifications
- Final evaluation on test set

### Training Time Estimates

- **CPU**: ~4-6 hours for 4 epochs (70K training samples)
- **GPU (CUDA)**: ~30-60 minutes for 4 epochs (70K training samples)
- **GPU (M1/M2 Mac)**: ~1-2 hours for 4 epochs

### GPU vs CPU Considerations

**GPU (Recommended)**:
- Much faster training (10-20x speedup)
- Required for large batch sizes
- Automatic mixed precision support

**CPU**:
- Slower but works for smaller datasets
- No special setup required
- Use smaller batch sizes (8 or 4)

To check GPU availability:
```python
import torch
print(torch.cuda.is_available())  # Should print True if GPU available
```

---

## Model Files and Outputs

### Directory Structure After Training

```
dl_methods/roberta/
├── model/
│   ├── best/              # Best model checkpoint (highest validation accuracy)
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   ├── final/             # Final model after all epochs
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   ├── tokenizer/         # RoBERTa tokenizer files
│   │   ├── vocab.json
│   │   ├── merges.txt
│   │   └── ...
│   └── label_encoder.pkl  # Label encoder (maps labels to indices)
├── logs/
│   ├── training_history.json      # Per-epoch metrics
│   ├── evaluation_log.json        # Comprehensive evaluation metrics
│   ├── evaluation_report.txt      # Classification report
│   └── hyperparameters.json      # Training configuration
└── roberta_pipeline_training_curve.png  # Training curves visualization
```

### Saved Files Description

1. **Model Checkpoints**:
   - `model/best/`: Best model based on validation accuracy (use for inference)
   - `model/final/`: Final model after all epochs

2. **Tokenizer** (`model/tokenizer/`):
   - Vocabulary and tokenization rules
   - Required for text preprocessing during inference

3. **Label Encoder** (`model/label_encoder.pkl`):
   - Maps label names to indices (0 = Great/Safe, 1 = Bait/Phishing)
   - Required for decoding predictions

4. **Training History** (`logs/training_history.json`):
   - Per-epoch training and validation metrics
   - Useful for analyzing training progress

5. **Evaluation Logs** (`logs/evaluation_log.json`):
   - Comprehensive metrics: accuracy, precision, recall, F1-score
   - Test set performance

6. **Evaluation Report** (`logs/evaluation_report.txt`):
   - Human-readable classification report
   - Per-class metrics

### Locating and Using Saved Models

**Best model path** (recommended for inference):
```python
BEST_MODEL_PATH = "dl_methods/roberta/model/best"
TOKENIZER_PATH = "dl_methods/roberta/model/tokenizer"
LABEL_ENCODER_PATH = "dl_methods/roberta/model/label_encoder.pkl"
```

**Final model path**:
```python
FINAL_MODEL_PATH = "dl_methods/roberta/model/final"
```

---

## Inference

### Loading Trained Models

#### Method 1: Using the Inference Script

The project includes a dedicated inference script for RoBERTa:

```bash
python 09_inference.py
```

This script:
- Loads the best RoBERTa model
- Processes the demonstration dataset
- Saves predictions to `dl_methods/roberta/logs/demonstration_dataset_predictions.json`

#### Method 2: Programmatic Loading

```python
import torch
import pickle
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Paths
MODEL_DIR = "dl_methods/roberta"
BEST_MODEL_PATH = f"{MODEL_DIR}/model/best"
TOKENIZER_PATH = f"{MODEL_DIR}/model/tokenizer"
LABEL_ENCODER_PATH = f"{MODEL_DIR}/model/label_encoder.pkl"

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

# Load label encoder
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Load model
num_classes = len(label_encoder.classes_)
model = RobertaForSequenceClassification.from_pretrained(
    BEST_MODEL_PATH,
    num_labels=num_classes,
    local_files_only=True
)

# Set to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
```

### Single Email Prediction

```python
def predict_email(model, tokenizer, text, max_len=128):
    """Predict if an email is phishing."""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    # Decode label
    label_name = label_encoder.inverse_transform([pred_idx])[0]
    
    return {
        'prediction': label_name,
        'confidence': confidence,
        'probabilities': {
            'Great': probs[0][0].item(),
            'Bait': probs[0][1].item()
        }
    }

# Example usage
email_text = "Your account has been suspended. Click here to verify..."
result = predict_email(model, tokenizer, email_text)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### Batch Prediction

```python
from torch.utils.data import Dataset, DataLoader

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def predict_batch(model, tokenizer, texts, batch_size=32):
    """Predict on a batch of emails."""
    dataset = InferenceDataset(texts, tokenizer, max_len=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            _, preds = torch.max(logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    return all_predictions, all_probabilities

# Example usage
emails = [
    "Your account has been suspended...",
    "Meeting reminder for tomorrow...",
    "Congratulations! You've won a prize..."
]
predictions, probabilities = predict_batch(model, tokenizer, emails)
```

### Using with 09_inference.py

The `09_inference.py` script is specifically designed for RoBERTa inference:

```bash
python 09_inference.py
```

**Input**: `datasets/Demonstration dataset.csv` (must have 'Title' and 'Text' columns)

**Output**: `dl_methods/roberta/logs/demonstration_dataset_predictions.json`

The output includes:
- Individual predictions for each sample
- Confidence scores and class probabilities
- Summary statistics (accuracy, prediction distribution)

### Using with streamlit_app.py

The Streamlit app uses RoBERTa for real-time inference:

```bash
streamlit run streamlit_app.py
```

Features:
- Interactive web interface
- Real-time predictions
- Confidence scores
- Model performance metrics

### Output Format

Predictions return:
- **Prediction**: Class label (Great/Safe or Bait/Phishing)
- **Confidence**: Probability of predicted class (0-1)
- **Probabilities**: Probability distribution over all classes

Example output:
```python
{
    'prediction': 'Bait',
    'confidence': 0.9876,
    'probabilities': {
        'Great': 0.0124,
        'Bait': 0.9876
    }
}
```

---

## Configuration

### Default Hyperparameters

```python
{
    'model_name': 'roberta-base',
    'max_len': 128,
    'batch_size': 16,
    'epochs': 4,
    'learning_rate': 2e-5,
    'random_seed': 42,
    'optimizer': 'AdamW',
    'loss_function': 'CrossEntropyLoss with class weights',
    'scheduler': 'LinearScheduleWithWarmup',
    'mixed_precision': True
}
```

### Customizing Training Parameters

Modify the pipeline initialization:

```python
from dl_methods.roberta_pipeline import RobertaPipeline

# Custom configuration
pipeline = RobertaPipeline(
    batch_size=32,        # Larger batch size (requires more GPU memory)
    epochs=6,             # More epochs
    learning_rate=3e-5,   # Slightly higher learning rate
    max_len=256,          # Longer sequences
    random_seed=123       # Different random seed
)
```

### Model-Specific Settings

- **Base Model**: `roberta-base` (125M parameters)
- **Tokenization**: RoBERTa tokenizer (BPE-based)
- **Special Tokens**: `<s>`, `</s>`, `<pad>`, `<unk>`
- **Vocabulary Size**: 50,265 tokens

### Environment Variables

No special environment variables required. The model automatically detects:
- GPU availability (`CUDA_VISIBLE_DEVICES` respected)
- Device selection (CPU/GPU)

---

## Examples

### Complete Training Example

```python
from dl_methods.roberta_pipeline import RobertaPipeline

# Initialize pipeline
pipeline = RobertaPipeline(
    batch_size=16,
    epochs=4,
    learning_rate=2e-5
)

# Run training
history = pipeline.run(
    train_path="cleaned_data/DL/train/train_split.csv",
    val_path="cleaned_data/DL/validation/validation_split.csv",
    test_path="cleaned_data/DL/test/test_split.csv"
)

# Access training history
print(f"Final training accuracy: {history['train_acc'][-1]}")
print(f"Final validation accuracy: {history['val_acc'][-1]}")
```

### Complete Inference Example

```python
import torch
import pickle
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Load model components
MODEL_DIR = "dl_methods/roberta"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = RobertaTokenizer.from_pretrained(
    f"{MODEL_DIR}/model/tokenizer", 
    local_files_only=True
)

with open(f"{MODEL_DIR}/model/label_encoder.pkl", 'rb') as f:
    label_encoder = pickle.load(f)

model = RobertaForSequenceClassification.from_pretrained(
    f"{MODEL_DIR}/model/best",
    num_labels=len(label_encoder.classes_),
    local_files_only=True
)
model.to(device)
model.eval()

# Predict
email = "Your account will be closed. Verify now!"
inputs = tokenizer(email, return_tensors="pt", truncation=True, 
                   padding=True, max_length=128).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()

label = label_encoder.inverse_transform([pred_idx])[0]
confidence = probs[0][pred_idx].item()

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.4f}")
```

### Integration with Other Scripts

**Model Interpretation**:
```bash
# First run inference
python 09_inference.py

# Then generate attention visualizations
python 10_model_interpretation.py
```

**Results Aggregation**:
```bash
# Aggregate all model results (includes RoBERTa)
python 11_aggregate_results.py
```

### Common Use Cases

1. **Training from scratch**: Use default parameters
2. **Fine-tuning existing model**: Load pretrained and continue training
3. **Batch processing**: Use `09_inference.py` or custom batch prediction
4. **Real-time inference**: Use `streamlit_app.py` or programmatic loading
5. **Model evaluation**: Check `logs/evaluation_log.json` for metrics

---

## Troubleshooting

### Common Errors and Solutions

#### 1. Model Not Found Error

**Error**: `FileNotFoundError: Model not found at dl_methods/roberta/model/best`

**Solution**: 
- Ensure training has completed successfully
- Check that model files exist in `dl_methods/roberta/model/`
- If using final model instead: change path to `model/final`

#### 2. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size: `batch_size=8` or `batch_size=4`
- Reduce max sequence length: `max_len=64`
- Use gradient accumulation (modify base_pipeline.py)
- Train on CPU (slower but works)

#### 3. Tokenizer Not Found

**Error**: `FileNotFoundError: Tokenizer not found`

**Solution**:
- Ensure tokenizer files exist in `dl_methods/roberta/model/tokenizer/`
- Check that training saved tokenizer correctly
- Try using `model/final` directory if tokenizer not in separate directory

#### 4. Data Format Errors

**Error**: `KeyError: 'text'` or `KeyError: 'label'`

**Solution**:
- Ensure CSV files have 'text' and 'label' columns
- Check column names match exactly (case-sensitive)
- Verify data preprocessing completed successfully

#### 5. Memory Issues (CPU)

**Error**: Slow training or system freezing

**Solutions**:
- Reduce batch size to 4 or 8
- Use smaller dataset subset for testing
- Close other applications
- Consider using GPU if available

#### 6. Import Errors

**Error**: `ModuleNotFoundError: No module named 'transformers'`

**Solution**:
```bash
pip install -r requirements.txt
```

#### 7. Label Encoder Issues

**Error**: `AttributeError: 'LabelEncoder' object has no attribute 'classes_'`

**Solution**:
- Ensure label encoder was saved correctly during training
- Re-train model if label encoder is corrupted
- Check pickle file is not corrupted

### Model Loading Issues

If model fails to load:
1. Verify all files exist in model directory
2. Check file permissions
3. Ensure correct model path
4. Try loading from `model/final` instead of `model/best`

### CUDA/GPU Problems

**Check GPU availability**:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

**Force CPU usage**:
```python
device = torch.device('cpu')
model.to(device)
```

### Data Format Errors

Ensure CSV files have:
- Header row with 'text' and 'label' columns
- Proper encoding (UTF-8)
- No empty rows
- Valid label values (0 or 1)

---

## Performance

### Expected Performance Metrics

Based on project evaluations, RoBERTa typically achieves:

- **Test Accuracy**: ~98-99%
- **Weighted Precision**: ~98-99%
- **Weighted Recall**: ~98-99%
- **F1-Score**: ~98-99%

*Note: Actual performance depends on dataset quality and size*

### Model Comparison Notes

RoBERTa outperforms:
- BERT (similar architecture, but RoBERTa is better optimized)
- DistilBERT (smaller, faster, but lower accuracy)
- Traditional ML models (Logistic Regression, Naive Bayes, SVM)

**Why RoBERTa performs best**:
- Better pretraining strategy
- More robust to different text patterns
- Handles phishing-specific features well

### Optimization Tips

1. **Use GPU**: 10-20x speedup for training
2. **Batch size**: Larger batches (if memory allows) for faster training
3. **Mixed precision**: Already enabled, reduces memory usage
4. **Sequence length**: Use 128 for balance (256/512 slower but may improve accuracy)
5. **Early stopping**: Monitor validation loss to prevent overfitting

---

## References

### Relevant Scripts

- **Training Pipeline**: [../roberta_pipeline.py](../roberta_pipeline.py)
- **Base Pipeline**: [../base_pipeline.py](../base_pipeline.py)
- **Inference Script**: [../../09_inference.py](../../09_inference.py)
- **Model Interpretation**: [../../10_model_interpretation.py](../../10_model_interpretation.py)
- **Data Preprocessing**: [../../03_dl_preprocessing_eda.py](../../03_dl_preprocessing_eda.py)
- **Submission Helper**: [../../submission_helper.py](../../submission_helper.py)
- **Streamlit App**: [../../streamlit_app.py](../../streamlit_app.py)

### Main Project README

For general project information, see: [../../README.md](../../README.md)

### Model Documentation

- **RoBERTa Paper**: [Liu et al., 2019](https://arxiv.org/abs/1907.11692)
- **Hugging Face RoBERTa**: [https://huggingface.co/roberta-base](https://huggingface.co/roberta-base)
- **Transformers Documentation**: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

### Additional Resources

- PyTorch Documentation: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- Hugging Face Course: [https://huggingface.co/course](https://huggingface.co/course)

---

## Support

For issues or questions:
- Check the main project README: [../../README.md](../../README.md)
- Review troubleshooting section above
- Check model logs in `dl_methods/roberta/logs/`

**Project**: MIE1517 Fall 2025, University of Toronto - Group 11

