import requests
from bs4 import BeautifulSoup
import nltk
import pandas as pd
import os
from datetime import datetime

nltk.download("punkt")

# Define folders
RAW_DATA_PATH = "./data/raw/"
os.makedirs(RAW_DATA_PATH, exist_ok=True)

# BBC News URL
BBC_NEWS_URL = "https://www.bbc.com/news"

# Headers to mimic a real browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Function to extract article links
def get_article_links(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()

        for a_tag in soup.find_all("a", href=True):
            link = a_tag["href"]
            if link.startswith("/news") and "/live/" not in link:  # Avoid live news
                links.add(f"https://www.bbc.com{link}")

        return list(links)[:10]  # Get top 10 articles
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching BBC News page: {e}")
        return []

# Function to scrape text from an article
def extract_sentences(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract text from paragraphs or fallback to divs
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        if not paragraphs:
            paragraphs = [div.get_text() for div in soup.find_all("div")]

        text = " ".join(paragraphs)
        sentences = nltk.sent_tokenize(text)

        # Filter out garbage sentences
        sentences = [s for s in sentences if len(s.split()) > 5]  # At least 5 words
        return sentences
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching {url}: {e}")
        return []

# Main scraping logic
article_links = get_article_links(BBC_NEWS_URL)
all_sentences = []

for link in article_links:
    all_sentences.extend(extract_sentences(link))

# Keep only the first 200 sentences
all_sentences = all_sentences[:200]

# Save raw data with timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
raw_filename = f"bbc_sentences_raw_{timestamp}.csv"
raw_data_path = os.path.join(RAW_DATA_PATH, raw_filename)

df = pd.DataFrame({"sentence": all_sentences})
df.to_csv(raw_data_path, index=False)

print(f"✅ Scraping complete. Saved {len(all_sentences)} sentences to {raw_data_path}")

# Check if file was actually created
if os.path.exists(raw_data_path):
    print("✅ File saved successfully!")
else:
    print("❌ File NOT found! Check script permissions or file path.")
