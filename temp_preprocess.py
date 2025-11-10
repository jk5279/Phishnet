import os
import pandas as pd
from pathlib import Path

# Configuration
TEST_DATASETS_DIR = "datasets/raw - DO NOT OVERWRITE/test_datasets"

# Label standardization mapping
LABEL_MAP = {
    "Phishing": 1,
    "phishing": 1,
    "PHISHING": 1,
    "1": 1,
    "1.0": 1,
    1: 1,
    "Not Phishing": 0,
    "not phishing": 0,
    "NOT PHISHING": 0,
    "Safe Email": 0,
    "safe email": 0,
    "0": 0,
    "0.0": 0,
    0: 0,
    "ham": 0,
    "spam": 1,
}

def preprocess_file(filepath):
    print(f"\nProcessing: {os.path.basename(filepath)}")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    
    print(f"  Original columns: {list(df.columns)}")
    print(f"  Original shape: {df.shape}")
    
    text_col = None
    label_col = None
    
    text_candidates = ['text', 'body', 'Text', 'Body', 'email', 'Email']
    for col in df.columns:
        if col.lower() in [c.lower() for c in text_candidates]:
            text_col = col
            break
    
    label_candidates = ['label', 'phishing_flag', 'Label', 'Phishing Flag', 'Class', 'class']
    for col in df.columns:
        if col.lower() in [c.lower() for c in label_candidates]:
            label_col = col
            break
    
    if text_col is None:
        raise ValueError(f"Could not find text column. Available: {list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Could not find label column. Available: {list(df.columns)}")
    
    print(f"  Using text column: '{text_col}'")
    print(f"  Using label column: '{label_col}'")
    
    result_df = pd.DataFrame()
    result_df['text'] = df[text_col].astype(str)
    result_df['label'] = df[label_col]
    
    print("  Standardizing labels...")
    original_labels = result_df['label'].unique()
    print(f"  Original label values: {original_labels}")
    
    def map_label(value):
        if pd.isna(value):
            return None
        value_str = str(value).strip()
        if value_str in LABEL_MAP:
            return LABEL_MAP[value_str]
        value_lower = value_str.lower()
        for key, mapped_value in LABEL_MAP.items():
            if str(key).lower() == value_lower:
                return mapped_value
        if "phish" in value_lower or value_str == "1":
            return 1
        return 0
    
    result_df['label'] = result_df['label'].apply(map_label)
    
    before_count = len(result_df)
    result_df = result_df.dropna(subset=['text', 'label'])
    after_count = len(result_df)
    if before_count != after_count:
        print(f"  Removed {before_count - after_count} rows with missing values")
    
    result_df = result_df[result_df['text'].str.strip().str.len() > 0]
    result_df['label'] = result_df['label'].astype(int)
    
    final_labels = result_df['label'].unique()
    print(f"  Final label values: {sorted(final_labels)}")
    print(f"  Label distribution: {result_df['label'].value_counts().to_dict()}")
    print(f"  Final shape: {result_df.shape}")
    
    return result_df

# Main execution
print("=" * 60)
print("Test Datasets Preprocessing")
print("=" * 60)

test_dir = Path(TEST_DATASETS_DIR)
if not test_dir.exists():
    print(f"Error: Directory not found: {TEST_DATASETS_DIR}")
else:
    csv_files = list(test_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {TEST_DATASETS_DIR}")
    else:
        print(f"\nFound {len(csv_files)} CSV file(s) to process:")
        for f in csv_files:
            print(f"  - {f.name}")
        
        for csv_file in csv_files:
            try:
                preprocessed_df = preprocess_file(str(csv_file))
                output_filename = csv_file.parent / f"{csv_file.stem}_preprocessed.csv"
                preprocessed_df.to_csv(output_filename, index=False, encoding='utf-8')
                print(f"  ✓ Saved: {output_filename.name}")
            except Exception as e:
                print(f"  ✗ Error processing {csv_file.name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("Preprocessing complete!")
        print("=" * 60)

