"""
DistilBERT Pipeline.
Train and evaluate DistilBERT model for phishing detection.
"""

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from .base_pipeline import BaseDLPipeline
from .utils import get_pretrained_path

class DistilBertPipeline(BaseDLPipeline):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="distilbert_pipeline",
            pretrained_model_name="distilbert-base-cased",
            **kwargs
        )
        
    def _create_model(self, num_classes: int):
        model_path = get_pretrained_path(self.pretrained_model_name)
        print(f"Loading DistilBERT from {model_path}...")
        return DistilBertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_classes,
            local_files_only=True
        )
        
    def _get_tokenizer(self):
        model_path = get_pretrained_path(self.pretrained_model_name)
        return DistilBertTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

if __name__ == "__main__":
    pipeline = DistilBertPipeline(
        batch_size=16,
        epochs=4,
        learning_rate=2e-5
    )
    
    pipeline.run(
        train_path="cleaned_data/train_split.csv",
        val_path="cleaned_data/validation_split.csv",
        test_path="cleaned_data/test_split.csv"
    )

