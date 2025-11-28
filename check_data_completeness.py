"""
check_data_completeness.py

Check all split CSV files for missing labels, source_file, or source_name.
Also identifies rows with extremely long text that might cause CSV viewer issues.
"""

import csv
import sys

csv.field_size_limit(sys.maxsize)

FILES_TO_CHECK = [
    "cleaned_data/train_split.csv",
    "cleaned_data/validation_split.csv",
    "cleaned_data/test_split.csv",
]

def check_file(filename):
    """Check a single CSV file for data completeness issues."""
    print(f"\n{'='*80}")
    print(f"Checking: {filename}")
    print('='*80)
    
    issues = {
        'empty_label': [],
        'empty_source_file': [],
        'empty_source_name': [],
        'very_long_text': [],  # >100k chars
        'extremely_long_text': [],  # >1M chars
    }
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            print(f"Total rows: {len(rows)}")
            
            for i, row in enumerate(rows):
                label = str(row.get('label', '')).strip()
                source_file = str(row.get('source_file', '')).strip()
                source_name = str(row.get('source_name', '')).strip()
                text = str(row.get('text', ''))
                text_len = len(text)
                
                # Check for empty values
                if not label or label == '' or label.lower() in ['nan', 'none', 'null']:
                    issues['empty_label'].append(i+1)
                
                if not source_file or source_file == '' or source_file.lower() in ['nan', 'none', 'null']:
                    issues['empty_source_file'].append(i+1)
                
                if not source_name or source_name == '' or source_name.lower() in ['nan', 'none', 'null']:
                    issues['empty_source_name'].append(i+1)
                
                # Check for very long text
                if text_len > 1000000:  # >1M chars
                    issues['extremely_long_text'].append((i+1, text_len, source_file))
                elif text_len > 100000:  # >100k chars
                    issues['very_long_text'].append((i+1, text_len, source_file))
            
            # Report findings
            print(f"\nData Completeness:")
            print(f"  Empty labels: {len(issues['empty_label'])}")
            if issues['empty_label']:
                print(f"    Row numbers: {issues['empty_label'][:10]}{'...' if len(issues['empty_label']) > 10 else ''}")
            
            print(f"  Empty source_file: {len(issues['empty_source_file'])}")
            if issues['empty_source_file']:
                print(f"    Row numbers: {issues['empty_source_file'][:10]}{'...' if len(issues['empty_source_file']) > 10 else ''}")
            
            print(f"  Empty source_name: {len(issues['empty_source_name'])}")
            if issues['empty_source_name']:
                print(f"    Row numbers: {issues['empty_source_name'][:10]}{'...' if len(issues['empty_source_name']) > 10 else ''}")
            
            print(f"\nText Length Issues (may cause CSV viewer problems):")
            print(f"  Extremely long text (>1M chars): {len(issues['extremely_long_text'])}")
            if issues['extremely_long_text']:
                print(f"    These rows may cause CSV viewers to crash or misparse:")
                for row_num, text_len, source_file in issues['extremely_long_text'][:5]:
                    print(f"      Row {row_num}: {text_len:,} chars, source_file={source_file}")
            
            print(f"  Very long text (>100k chars): {len(issues['very_long_text'])}")
            if issues['very_long_text']:
                print(f"    Sample rows:")
                for row_num, text_len, source_file in issues['very_long_text'][:5]:
                    print(f"      Row {row_num}: {text_len:,} chars, source_file={source_file}")
            
            # Summary
            total_issues = (len(issues['empty_label']) + 
                          len(issues['empty_source_file']) + 
                          len(issues['empty_source_name']))
            
            if total_issues == 0:
                print(f"\n✓ All rows have complete data (labels, source_file, source_name)")
            else:
                print(f"\n⚠️  Found {total_issues} rows with missing data")
            
            if issues['extremely_long_text']:
                print(f"\n⚠️  WARNING: {len(issues['extremely_long_text'])} rows have extremely long text")
                print(f"    These may cause CSV viewer display issues.")
                print(f"    The data is complete, but viewers may not display it correctly.")
            
            return issues
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("="*80)
    print("Data Completeness Check")
    print("="*80)
    
    all_issues = {}
    for filename in FILES_TO_CHECK:
        issues = check_file(filename)
        if issues:
            all_issues[filename] = issues
    
    # Overall summary
    print("\n" + "="*80)
    print("Overall Summary")
    print("="*80)
    
    total_empty_labels = sum(len(issues.get('empty_label', [])) for issues in all_issues.values())
    total_empty_source_file = sum(len(issues.get('empty_source_file', [])) for issues in all_issues.values())
    total_extremely_long = sum(len(issues.get('extremely_long_text', [])) for issues in all_issues.values())
    
    if total_empty_labels == 0 and total_empty_source_file == 0:
        print("✓ All files have complete data - no missing labels or source_file values")
    else:
        print(f"⚠️  Found missing data:")
        print(f"  Empty labels: {total_empty_labels}")
        print(f"  Empty source_file: {total_empty_source_file}")
    
    if total_extremely_long > 0:
        print(f"\n⚠️  {total_extremely_long} rows have extremely long text (>1M chars)")
        print("   These may cause CSV viewer display issues.")
        print("   Recommendation: Use a CSV viewer that handles large fields, or")
        print("   use Python/pandas to read the files programmatically.")

