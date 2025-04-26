#!/usr/bin/env python3
import re

def update_urls_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Replace URLs in the file
    content = content.replace('https://api.filot.io/v1', 'https://filotanalytics.replit.app/v1')
    
    with open(file_path, 'w') as file:
        file.write(content)
    
    print(f'Updated URLs in {file_path}')

# Update both files
update_urls_in_file('api_documentation.md')
update_urls_in_file('pages/5_API_Documentation.py')