# image_scraper.py

import os
import requests
from duckduckgo_search import DDGS

# --- Configuration ---
BASE_DIR = "data/fragility"
MAX_IMAGES_PER_CLASS = 500
MAX_RESULTS_PER_QUERY = 100

SEARCH_QUERIES = {
    "fragile": [
        "glass vase", "ceramic plate", "mirror", "wine glass", "porcelain figurine",
        "tablet screen", "light bulb", "glass bottle", "fragile electronics"
    ],
    "non_fragile": [
        "t-shirt", "book", "plastic container", "wooden box", "backpack",
        "notebook", "shoe", "fabric roll", "cardboard box"
    ]
}

def download_image(url, save_path):
    try:
        img_data = requests.get(url, timeout=10).content
        with open(save_path, 'wb') as f:
            f.write(img_data)
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def scrape_images():
    with DDGS() as ddgs:
        for label, queries in SEARCH_QUERIES.items():
            label_dir = os.path.join(BASE_DIR, label)
            os.makedirs(label_dir, exist_ok=True)
            count = 0
            seen_urls = set()

            for query in queries:
                if count >= MAX_IMAGES_PER_CLASS:
                    break

                print(f"🔍 Searching for '{query}'...")
                results = ddgs.images(keywords=query, max_results=MAX_RESULTS_PER_QUERY)

                for result in results:
                    if count >= MAX_IMAGES_PER_CLASS:
                        break

                    image_url = result.get("image")
                    if not image_url or image_url in seen_urls:
                        continue

                    seen_urls.add(image_url)
                    filename = f"{label}_{count}.jpg"
                    save_path = os.path.join(label_dir, filename)

                    if download_image(image_url, save_path):
                        count += 1
                        print(f"✅ Saved: {filename}")

            print(f"📁 Finished downloading {count} images for class '{label}'")

if __name__ == "__main__":
    scrape_images()