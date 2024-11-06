# بسم الله الرحمن الرحيم و به نستعين

import os
import requests
from bs4 import BeautifulSoup

url = 'https://example.com/'  # Replace with the URL of the website containing the Excel files
login_url = 'https://example.com/login'  # Replace with the URL of the login page or endpoint
download_folder = 'downloaded_files'  # Folder where you want to save the downloaded files

# Replace with your email and password
email = 'your_email@example.com'
password = 'your_password'

# Start a session to maintain cookies between requests
session = requests.Session()

# Authenticate with the website
login_data = {
    'email': email,
    'password': password
    # Add other form fields if necessary
}
response = session.post(login_url, data=login_data)
response.raise_for_status()

# Fetch the content of the web page
response = session.get(url)
response.raise_for_status()

# Parse the content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all links to Excel files
excel_links = soup.find_all('a', href=lambda href: href and href.endswith('.xlsx'))

# Create the download folder if it doesn't exist
os.makedirs(download_folder, exist_ok=True)

# Download the Excel files
for link in excel_links:
    file_url = link['href']
    file_name = os.path.join(download_folder, os.path.basename(file_url))

    if not file_url.startswith('http'):
        file_url = url.rstrip('/') + '/' + file_url.lstrip('/')

    print(f'Downloading {file_url}...')
    response = session.get(file_url)
    response.raise_for_status()

    with open(file_name, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

print('All Excel files have been downloaded.')
