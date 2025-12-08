# BERT Model Guide

## Overview

This guide provides comprehensive instructions for training and using the **BERT (Bidirectional Encoder Representations from Transformers)** model for phishing email detection. BERT is a powerful transformer-based model that serves as an alternative to RoBERTa.

### Model Architecture

- **Base Model**: `bert-base-cased` (110M parameters)
- **Task**: Binary sequence classification (Safe/Phishing)
- **Architecture**: Transformer-based bidirectional encoder with classification head
- **Input**: Email text (max 128 tokens)
- **Output**: Binary classification (Great/Safe = 0, Bait/Phishing = 1)

### Key Features

- **Bidirectional Context**: Processes text in both directions simultaneously
- **Case-Sensitive**: Uses cased tokenizer (preserves capitalization)
- **Mixed Precision Training**: Automatic mixed precision (AMP) for faster training
- **Class Weight Balancing**: Handles imbalanced datasets automatically
- **Comprehensive Logging**: Detailed training history and evaluation metrics
- **Model Checkpointing**: Saves both best and final model checkpoints

### BERT vs RoBERTa

**Similarities**:
- Same pipeline structure and training process
- Similar architecture (transformer encoder)
- Same data preprocessing requirements
- Compatible inference methods

**Differences**:
- **BERT**: 110M parameters, case-sensitive tokenizer
- **RoBERTa**: 125M parameters, optimized pretraining
- **Performance**: RoBERTa typically achieves slightly better accuracy
- **When to use BERT**: Good baseline, well-documented, widely used

**Recommendation**: For best performance, use RoBERTa. Use BERT if you need a well-established baseline or want to compare architectures.

### Related Files

- Main project README: [../../README.md](../../README.md)
- Training pipeline: [../bert_pipeline.py](../bert_pipeline.py)
- Base pipeline class: [../base_pipeline.py](../base_pipeline.py)
- RoBERTa guide: [../roberta/README.md](../roberta/README.md) (for comparison)

---

## Prerequisites

### Python Version

- Python 3.8 or higher

### Required Packages

Install all dependencies from the project root:

```bash
pip install -r requirements.txt
```

Key packages for BERT:
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
python -m dl_methods.bert_pipeline

# Or from dl_methods directory
cd dl_methods
python bert_pipeline.py
```

### Inference

```python
import torch
import pickle
from transformers import BertForSequenceClassification, BertTokenizer

# Load model
MODEL_DIR = "dl_methods/bert"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(
    f"{MODEL_DIR}/model/tokenizer", 
    local_files_only=True
)

with open(f"{MODEL_DIR}/model/label_encoder.pkl", 'rb') as f:
    label_encoder = pickle.load(f)

model = BertForSequenceClassification.from_pretrained(
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

print(f"Prediction: {label}, Confidence: {confidence:.4f}")
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
- Cleans text while preserving case and punctuation (important for BERT's cased tokenizer)
- Filters by length (5-2000 tokens)
- Removes duplicates
- Creates train/validation/test splits (70%/15%/15%)

**Note**: BERT uses a case-sensitive tokenizer (`bert-base-cased`), so preserving case is important.

---

## Training

### Step-by-Step Training Instructions

1. **Ensure data is prepared** (see Data Preparation section)

2. **Run the training pipeline**:

```bash
# From project root
python -m dl_methods.bert_pipeline
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
python bert_pipeline.py
```

### Configuration Options

You can customize training by modifying the pipeline initialization:

```python
from dl_methods.bert_pipeline import BertPipeline

pipeline = BertPipeline(
    batch_size=16,        # Batch size (default: 16)
    epochs=4,             # Number of training epochs (default: 4)
    learning_rate=2e-5,   # Learning rate (default: 2e-5)
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
dl_methods/bert/
├── model/
│   ├── best/              # Best model checkpoint (highest validation accuracy)
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   ├── final/             # Final model after all epochs
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   ├── tokenizer/         # BERT tokenizer files
│   │   ├── vocab.txt
│   │   ├── tokenizer_config.json
│   │   └── ...
│   └── label_encoder.pkl  # Label encoder (maps labels to indices)
├── logs/
│   ├── training_history.json      # Per-epoch metrics
│   ├── evaluation_log.json        # Comprehensive evaluation metrics
│   ├── evaluation_report.txt      # Classification report
│   └── hyperparameters.json        # Training configuration
└── bert_pipeline_training_curve.png  # Training curves visualization
```

### Saved Files Description

1. **Model Checkpoints**:
   - `model/best/`: Best model based on validation accuracy (use for inference)
   - `model/final/`: Final model after all epochs

2. **Tokenizer** (`model/tokenizer/`):
   - Vocabulary and tokenization rules (WordPiece tokenization)
   - Case-sensitive tokenizer for `bert-base-cased`
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
BEST_MODEL_PATH = "dl_methods/bert/model/best"
TOKENIZER_PATH = "dl_methods/bert/model/tokenizer"
LABEL_ENCODER_PATH = "dl_methods/bert/model/label_encoder.pkl"
```

**Final model path**:
```python
FINAL_MODEL_PATH = "dl_methods/bert/model/final"
```

---

## Inference

### Loading Trained Models

#### Programmatic Loading

```python
import torch
import pickle
from transformers import BertForSequenceClassification, BertTokenizer

# Paths
MODEL_DIR = "dl_methods/bert"
BEST_MODEL_PATH = f"{MODEL_DIR}/model/best"
TOKENIZER_PATH = f"{MODEL_DIR}/model/tokenizer"
LABEL_ENCODER_PATH = f"{MODEL_DIR}/model/label_encoder.pkl"

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

# Load label encoder
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Load model
num_classes = len(label_encoder.classes_)
model = BertForSequenceClassification.from_pretrained(
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
    'model_name': 'bert-base-cased',
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
from dl_methods.bert_pipeline import BertPipeline

# Custom configuration
pipeline = BertPipeline(
    batch_size=32,        # Larger batch size (requires more GPU memory)
    epochs=6,             # More epochs
    learning_rate=3e-5,   # Slightly higher learning rate
    max_len=256,          # Longer sequences
    random_seed=123       # Different random seed
)
```

### Model-Specific Settings

- **Base Model**: `bert-base-cased` (110M parameters)
- **Tokenization**: BERT WordPiece tokenizer (case-sensitive)
- **Special Tokens**: `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `[MASK]`
- **Vocabulary Size**: 28,996 tokens
- **Case Sensitivity**: Uses cased tokenizer (preserves capitalization)

### Environment Variables

No special environment variables required. The model automatically detects:
- GPU availability (`CUDA_VISIBLE_DEVICES` respected)
- Device selection (CPU/GPU)

---

## Examples

### Complete Training Example

```python
from dl_methods.bert_pipeline import BertPipeline

# Initialize pipeline
pipeline = BertPipeline(
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
from transformers import BertForSequenceClassification, BertTokenizer

# Load model components
MODEL_DIR = "dl_methods/bert"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(
    f"{MODEL_DIR}/model/tokenizer", 
    local_files_only=True
)

with open(f"{MODEL_DIR}/model/label_encoder.pkl", 'rb') as f:
    label_encoder = pickle.load(f)

model = BertForSequenceClassification.from_pretrained(
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
# Note: 10_model_interpretation.py is designed for RoBERTa
# For BERT, you can adapt the script or use similar attention visualization
```

**Results Aggregation**:
```bash
# Aggregate all model results (includes BERT)
python 11_aggregate_results.py
```

### Common Use Cases

1. **Training from scratch**: Use default parameters
2. **Fine-tuning existing model**: Load pretrained and continue training
3. **Batch processing**: Use custom batch prediction code
4. **Real-time inference**: Use programmatic loading
5. **Model evaluation**: Check `logs/evaluation_log.json` for metrics
6. **Comparison with RoBERTa**: Train both and compare performance

---

## Troubleshooting

### Common Errors and Solutions

#### 1. Model Not Found Error

**Error**: `FileNotFoundError: Model not found at dl_methods/bert/model/best`

**Solution**: 
- Ensure training has completed successfully
- Check that model files exist in `dl_methods/bert/model/`
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
- Ensure tokenizer files exist in `dl_methods/bert/model/tokenizer/`
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

Based on project evaluations, BERT typically achieves:

- **Test Accuracy**: ~96-98%
- **Weighted Precision**: ~96-98%
- **Weighted Recall**: ~96-98%
- **F1-Score**: ~96-98%

*Note: Actual performance depends on dataset quality and size. RoBERTa typically performs slightly better.*

### Model Comparison Notes

BERT vs RoBERTa:
- **BERT**: Well-established baseline, 110M parameters, case-sensitive
- **RoBERTa**: Optimized pretraining, 125M parameters, typically 1-2% better accuracy
- **Use BERT**: For baseline comparison, well-documented reference
- **Use RoBERTa**: For best performance (recommended)

BERT vs Other Models:
- Outperforms traditional ML models (Logistic Regression, Naive Bayes, SVM)
- Similar performance to DistilBERT but with more parameters
- Good balance between performance and model size

### Optimization Tips

1. **Use GPU**: 10-20x speedup for training
2. **Batch size**: Larger batches (if memory allows) for faster training
3. **Mixed precision**: Already enabled, reduces memory usage
4. **Sequence length**: Use 128 for balance (256/512 slower but may improve accuracy)
5. **Early stopping**: Monitor validation loss to prevent overfitting
6. **Case sensitivity**: BERT's cased tokenizer can help with proper nouns and capitalization patterns

---

## References

### Relevant Scripts

- **Training Pipeline**: [../bert_pipeline.py](../bert_pipeline.py)
- **Base Pipeline**: [../base_pipeline.py](../base_pipeline.py)
- **RoBERTa Guide**: [../roberta/README.md](../roberta/README.md) (for comparison)
- **Data Preprocessing**: [../../03_dl_preprocessing_eda.py](../../03_dl_preprocessing_eda.py)
- **Results Aggregation**: [../../11_aggregate_results.py](../../11_aggregate_results.py)

### Main Project README

For general project information, see: [../../README.md](../../README.md)

### Model Documentation

- **BERT Paper**: [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- **Hugging Face BERT**: [https://huggingface.co/bert-base-cased](https://huggingface.co/bert-base-cased)
- **Transformers Documentation**: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

### Additional Resources

- PyTorch Documentation: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- Hugging Face Course: [https://huggingface.co/course](https://huggingface.co/course)
- BERT Explained: [https://jalammar.github.io/illustrated-bert/](https://jalammar.github.io/illustrated-bert/)

---

## Support

For issues or questions:
- Check the main project README: [../../README.md](../../README.md)
- Review troubleshooting section above
- Check model logs in `dl_methods/bert/logs/`
- Compare with RoBERTa guide: [../roberta/README.md](../roberta/README.md)

**Project**: MIE1517 Fall 2025, University of Toronto - Group 11

