# --- Section 1: All Imports ---
import pandas as pd
import numpy as np
from collections import Counter
import warnings
import torch
import torch.nn as nn
from tqdm.auto import tqdm  # For progress bars

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score

# PyTorch imports
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW  # <-- CORRECTED import for AdamW
from torch.cuda.amp import autocast, GradScaler  # <-- CORRECTED import line

# Transformers imports
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

# --- Setup ---
warnings.filterwarnings('ignore')  # Suppress warnings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Running on device: {device} ---")

# --- Section 2: Constants ---
MODEL_NAME = '/scratch/kimjong/MIE1517-Fall2025-Group11/local_bert_cased'
MAX_LEN = 512
# <-- START WITH A SMALL BATCH SIZE (like 8 or 4) TO AVOID MEMORY CRASHES
BATCH_SIZE = 8
N_EPOCHS = 3
# This is the file created by your *first* preprocessing script
DATA_FILE = "/scratch/kimjong/MIE1517-Fall2025-Group11/cleaned_data/dl_dataset_final.csv"

# --- Section 3: PyTorch Dataset Class ---


class PhishingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# --- Section 4: Data Loading & Preprocessing ---
# THIS IS THE SECTION THAT WAS MISSING. IT DEFINES 'le'.
# =================================================================
print("\n--- [Section 4] Starting Data Loading & Preprocessing ---")

# 1. Load PRE-CLEANED data
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"ERROR: Could not find file {DATA_FILE}")
    print("Please make sure you have run the data cleaning pipeline (the script you showed me) first.")
    # Stop execution if file isn't found
    raise

print(f"Loaded pre-cleaned dataset: {df.shape}")

# 2. Encode Labels
#
#  HERE IS WHERE 'le' IS CREATED
#
le = LabelEncoder()
#
#
df['label_encoded'] = le.fit_transform(df['label'])
print(f"Label mapping: {list(zip(le.classes_, le.transform(le.classes_)))}")
NUM_CLASSES = len(le.classes_)  # 'le' is now defined, so this line will work

# 3. Prepare data for splitting
X = df['text']
y = df['label_encoded'].values

# 4. Step B4: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

# 5. Step B5: Tokenization & Dataset Creation
print(f"Initializing tokenizer ({MODEL_NAME})...")
tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME)  # 'tokenizer' is defined here

print("Creating PyTorch Datasets...")
train_dataset = PhishingDataset(
    texts=X_train.tolist(),
    labels=y_train,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
test_dataset = PhishingDataset(
    texts=X_test.tolist(),
    labels=y_test,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# 6. Step B6: Handling Class Imbalance (WeightedSampler)
print("Setting up WeightedRandomSampler for training...")
class_counts = np.bincount(y_train)
class_weights_per_sample = 1. / class_counts
sample_weights = np.array([class_weights_per_sample[t] for t in y_train])
sample_weights_tensor = torch.from_numpy(sample_weights).double()

sampler = WeightedRandomSampler(
    weights=sample_weights_tensor,
    num_samples=len(sample_weights_tensor),
    replacement=True
)

# 7. Create Final DataLoaders
# 'train_loader' and 'test_loader' are defined here
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=2
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# 8. (B6 Alternative) Calculate class weights for the loss function
class_weights_loss = compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
# 'class_weights_tensor' is defined here
class_weights_tensor = torch.tensor(
    class_weights_loss, dtype=torch.float).to(device)
print(f"Weights for loss function: {class_weights_tensor.cpu().numpy()}")
print("--- [Section 4] Data Loading Complete ---")
# =================================================================


# --- Section 5: Model Definition & Training Setup ---
# =================================================================
print(f"\n--- [Section 5] Starting Model Setup ---")
print(
    f"Loading model ({MODEL_NAME}) for {NUM_CLASSES}-class classification...")

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    output_attentions=False,
    output_hidden_states=False,
)
model.to(device)  # 'model' is defined here

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Scheduler
total_steps = len(train_loader) * N_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Loss Function (with weights)
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Mixed Precision Scaler
scaler = GradScaler()

# --- Section 6: Helper Functions (Training & Eval) ---


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training", unit="batch")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(data_loader)


def eval_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


# --- Section 7: Main Training Loop ---
print("\n--- [Section 7] Starting Training Loop ---")

for epoch in range(N_EPOCHS):
    print(f'\n--- Epoch {epoch + 1} / {N_EPOCHS} ---')

    avg_train_loss = train_epoch(
        model, train_loader, loss_fn, optimizer, device, scheduler, scaler
    )
    print(f'Average Training Loss: {avg_train_loss:.4f}')

    print("Running evaluation on test set...")
    labels, preds = eval_model(model, test_loader, device)

    accuracy = accuracy_score(labels, preds)
    print(f'Test Accuracy: {accuracy:.4f}')
    # Convert class names to strings for the report
    target_names_str = le.classes_.astype(str)
    print(classification_report(
        labels,
        preds,
        target_names=target_names_str
    ))

# --- Section 8: Single Prediction Function ---


def predict_single_string(text, model, tokenizer, label_encoder, device, max_len):
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    pred_index = torch.argmax(logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([pred_index])[0]

    return predicted_label, pred_index


# --- Section 9: Prediction Example ---
# This was fixed: string_var is now a string, not a list.
# =================================================================
print("\n--- [Section 9] Running Single Prediction ---")
string_var = """Tuition Payment Deadline Missed – Final Grace Period Mohit Malhotra<mohit.malhotra@mail.utoronto.ca> Dear Valued Student, This is your final warning. As of October 6, 2025, our records indicate that your Fall 2025 tuition deposit remains unpaid. You are now outside the formal payment deadline of September 30, 2025, and are at immediate risk of deregistration, loss of student status, and deactivation of all university services.Under the University of Toronto’s Financial Registration Policy, a minimum tuition deposit of $3,000–$5,000 is mandatory for all students, regardless of OSAP or external funding status. OSAP and other financial aid sources are disbursed after the term begins and do not replace the required initial deposit. To retain your course enrolment and active student status, you have been granted a final short-term grace period. Payment must now be submitted immediately using the instructions below: 🔹 Interac e-Transfer Details Email: Mabintydumbuya_19@hotmail.com Memo: Include your student number onlySecurity Question: Your name
Answer: Your student ID number⚠️ Do not send receipts to the transfer email address (Mabintydumbuya_19@hotmail.com)
All receipts and communications must be submitted via this official university correspondence channel only including the security questions and answers attached to your payment.
Failure to comply by the grace period will result in the following irreversible actions:
Immediate removal from all enrolled courses
Full deactivation of UTmail+, ACORN, Quercus, and all student systems
Placement of academic and financial holds, blocking transcript access, graduation eligibility, and future enrolment
Deregistration from the University of Toronto, with permanent implications for your academic record and immigration/visa status (if applicable)
If you are experiencing a genuine financial emergency, you must respond today with a detailed explanation and a formal request for urgent financial assistance. Failure to respond will be treated as non-compliance.
To resolve your status today, reply to this email with one of the following:
A copy of your Interac e-Transfer confirmation/receipt
A brief explanation for your non-payment and expected payment date
A formal request for emergency financial or legal aid
We cannot and will not hold your seat in courses without immediate compliance. This is your final opportunity to secure your enrolment and student status.
Best regards,
Mohit Malhotra
Office of the Bursar
Financial Aid & Awards University ofToronto"""

# This call will now work because 'model', 'tokenizer', 'le', 'device',
# and 'MAX_LEN' are all defined.
label, index = predict_single_string(
    string_var,
    model,
    tokenizer,
    le,
    device,
    MAX_LEN
)

print(f"Input text:    '{string_var[:100]}...'")  # Print first 100 chars
print(f"Predicted index: {index}")
print(f"Predicted label: '{label}'")

print("\n--- Script Finished ---")
