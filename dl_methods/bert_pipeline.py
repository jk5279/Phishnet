"""
BERT Pipeline.
Train and evaluate BERT model for phishing detection.
"""

from transformers import BertForSequenceClassification, BertTokenizer
from .base_pipeline import BaseDLPipeline
from .utils import get_pretrained_path

class BertPipeline(BaseDLPipeline):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="bert_pipeline",
            pretrained_model_name="bert-base-cased",
            **kwargs
        )
        
    def _create_model(self, num_classes: int):
        model_path = get_pretrained_path(self.pretrained_model_name)
        print(f"Loading BERT from {model_path}...")
        return BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_classes,
            local_files_only=True
        )
        
    def _get_tokenizer(self):
        model_path = get_pretrained_path(self.pretrained_model_name)
        return BertTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

if __name__ == "__main__":
    # Example usage
    pipeline = BertPipeline(
        batch_size=16,
        epochs=4,
        learning_rate=2e-5
    )
    
    # These paths assume standard directory structure from data aggregation
    pipeline.run(
        train_path="cleaned_data/train_split.csv",
        val_path="cleaned_data/validation_split.csv",
        test_path="cleaned_data/test_split.csv"
    )

