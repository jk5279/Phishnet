from transformers import BertTokenizer, BertForSequenceClassification
import os

MODEL_NAME = 'bert-base-cased'
# This will create a new folder named 'local_bert_cased'
# right inside your project directory
SAVE_PATH = os.path.join(os.getcwd(), 'local_bert_cased') 

print(f"Downloading model '{MODEL_NAME}' to '{SAVE_PATH}'...")

# Download and save tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(SAVE_PATH)

# Download and save model. 
# The num_labels is just a placeholder, it will be fine.
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.save_pretrained(SAVE_PATH)

print("Download and save complete.")
