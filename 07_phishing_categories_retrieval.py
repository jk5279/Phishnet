'''
Description:
Berkley has website detailing various phishing emails that students should be aware of
https://security.berkeley.edu/education-awareness/phishing/phishing-examples-archive
In this script I will scrape the website and retrieve the different catergory
For each category I will follow the link to get more details and try to retrieve the relevant information
This information will then be used as context for an LLM to generate synthetic phishing emails for training
'''

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urljoin
BASE_URL = "https://security.berkeley.edu/education-awareness/phishing/phishing-examples-archive"

# Step 1: Retrieve the main page and parse it for "Phishing Examples Archive" section
response = requests.get(BASE_URL)
print(f"Main page response status code: {response.status_code}")
soup = BeautifulSoup(response.content, 'html.parser')

# Find all views-row elements which contain phishing examples
phishing_examples = soup.find_all('div', class_='views-row')
phishing_data = []

for example in phishing_examples:
    # Find the link in the h2 tag
    link_element = example.find('h2').find('a')
    if link_element:
        relative_link = link_element.get('href')
        full_link = urljoin(BASE_URL, relative_link)
        title = link_element.text
        
        # Find the description in the paragraph using partial class matching
        description = ''
        # Look for any div that has 'field-name-body' as part of its class
        body_field = example.find('div', class_=lambda x: x and 'field-name-body' in x)
        if body_field:
            # Find any div that has 'field-item' as part of its class and contains a paragraph
            field_item = body_field.find('div', class_=lambda x: x and 'field-item' in x)
            if field_item:
                desc_element = field_item.find('p')
                if desc_element:
                    description = desc_element.text.strip()
                else:
                    # No paragraph found let's get the text directly
                    description = field_item.text.strip()
            # print(f"Found body field: {body_field is not None}, Found field item: {field_item is not None}")
        
        if not description:
            description = "No description available."
        
        phishing_data.append({
            'title': title,
            'link': full_link,
            'description': description
        })

print(f"Found {len(phishing_data)} phishing examples.")
time.sleep(1)
# for item in phishing_data:
#     print(f"Title: {item['title']}")
#     print(f"Link: {item['link']}")
#     print(f"Description: {item['description']}")
#     print("-" * 40)

# Step 2: We follow each link to get more details
for item in phishing_data:
    detail_response = requests.get(item['link'])
    detail_soup = BeautifulSoup(detail_response.content, 'html.parser')
    
    # Extract text for section called "What makes this a phishing message?"
    phishing_section = detail_soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], string="What makes this a phishing message?")
    if phishing_section:
        pishing_setcion_content = phishing_section.find_next('div').get_text(strip=True)
        # Sanity check if the content has the word "email"
        for i in range(4):
            if "email" not in pishing_setcion_content.lower():
                next_section = phishing_section.find_next()
                pishing_setcion_content = next_section.get_text(strip=True)
            else:
                break
        item['what_makes_this_phishing'] = pishing_setcion_content
        print(f"Found: {item['title']} \n What makes this phishing: {pishing_setcion_content}\n")
    
    # Wait a sec before the next request
    time.sleep(1)

# I will save this data to a CSV for further processing
df = pd.DataFrame(phishing_data)
df.to_csv('phishing_examples_berkley.csv', index=False)