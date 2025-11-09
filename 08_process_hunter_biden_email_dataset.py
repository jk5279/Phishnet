'''
Hunter Biden Email Dataset is quiet large (3GB) so it is not stored in this repo.
To get the original dataset please visit: https://www.kaggle.com/datasets/anuranroy/hunter-biden-mails

In this script I will open the json and extract the content of x emails and save them into a csv with
label set as "Not Phishing". This is not a phishing dataset so I am assuming they are all legitimate communications.

This csv will later be merged with other datasets and used for testing.
'''

import ijson
import os
import pandas as pd

dataset_path = '/mnt/c/Users/marcb/Documents/Uni/Masters/MIE1517/Project/Hunter Biden Email dataset/data.json'
csv_output_path = '/mnt/c/Users/marcb/Documents/Code Files/Phishnet/datasets/raw - DO NOT OVERWRITE/hunter_biden_emails'

def extract_emails(num_emails=1000):
    # Open the file in binary mode
    
    with open(dataset_path, 'rb') as f:
        parser = ijson.items(f, 'item')
        emails_to_save = []
        num_of_extracted = 0
        for _ in range(num_emails*100): # Read more items to account for skips
            # Get the first item
            entry = next(parser)
            raw_content = (entry['contents'])
            email_start_idx = -1
            # From raw_content I will look for a single email in the chain by looking for some keywords
            email_start_idx = raw_content.find("From:")
            if email_start_idx == -1:
                email_start_idx = raw_content.find("Sent:")
            if email_start_idx == -1:
                email_start_idx = raw_content.find("<html ")
            single_email = raw_content[:email_start_idx]
            # Sanity check email length
            if len(single_email) > 500:
                # Skip overly long emails
                continue
            num_of_extracted += 1
            emails_to_save.append({
                'text': single_email,
                'label': 'Not Phishing'
            })
            if num_of_extracted >= num_emails:
                break
    # Save to CSV
    df = pd.DataFrame(emails_to_save)
    os.makedirs(csv_output_path, exist_ok=True)
    output_file = os.path.join(csv_output_path, f'hunter_biden_{num_emails}_emails.csv')
    df.to_csv(output_file, index=False)

    print(f"Saved {num_emails} emails to {output_file}")

extract_emails(1000)