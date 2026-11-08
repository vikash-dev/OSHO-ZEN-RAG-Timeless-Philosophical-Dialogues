import os
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

output_folder = "Meditation"
os.makedirs(output_folder, exist_ok=True)

pdf_links = set()

try:
  # 1. Loop through pages 1 to 39 (range goes up to 40, since it's exclusive at the end)
  for page_num in range(1, 2):
    url = f"https://www.oshofragrance.org/books/search?booklanguages=2&bookauthors=Osho&bookcategories=Meditation&page={page_num}"
    print(f"Scraping page {page_num}: {url}")

    response = requests.get(url, headers=headers)

    # If a page doesn't exist or returns a 404, stop the loop gracefully
    if response.status_code != 200:
      print(f"Page {page_num} returned status {response.status_code}. Stopping.")
      break

    soup = BeautifulSoup(response.text, "html.parser")

    # Find all PDF links on the current page
    page_pdf_count = 0
    for link in soup.find_all("a", href=True):
      href = link["href"]
      if href.lower().endswith(".pdf"):
        absolute_url = urljoin(url, href)
        pdf_links.add(absolute_url)
        page_pdf_count += 1

    print(f"Found {page_pdf_count} PDFs on page {page_num}.")

    # Be polite to the server between page requests
    time.sleep(1)

  print(
      f"\nFinished scraping! Total unique PDF links found:"
      f" {len(pdf_links)}\n"
  )

  # 2. Download each unique PDF file
  for idx, pdf_url in enumerate(pdf_links):
    parsed_url = urlparse(pdf_url)
    filename = parsed_url.path.split("/")[-1] or f"document_{idx}.pdf"
    file_path = os.path.join(output_folder, filename)

    print(f"Downloading [{idx+1}/{len(pdf_links)}]: {filename}...")

    pdf_response = requests.get(pdf_url, headers=headers, stream=True)
    if pdf_response.status_code == 200:
      with open(file_path, "wb") as f:
        for chunk in pdf_response.iter_content(chunk_size=8192):
          f.write(chunk)
      print(f"Saved to: {file_path}")
    else:
      print(
          f"Failed to download {pdf_url} (Status: {pdf_response.status_code})"
      )

    time.sleep(1)

except Exception as e:
  print(f"An error occurred: {e}")