import os
import requests
import random
import time
from PIL import Image
from io import BytesIO

# --- Config ---
IMG_SIZE = (224, 224)
SAVE_ROOT = "data/booster_fragile"
NUM_IMAGES_PER_CATEGORY = 500
MAX_RETRIES = 10

# --- Categories and search terms ---
categories = {
    "glassware": "glass",
    "ceramics": "ceramic",
    "packaging": "fragile packaging",
    "mirrors": "mirror",
    "thin_materials": "light bulb"
}

# --- Helper: Save image from redirected URL ---
def fetch_unsplash_image(query):
    try:
        # Step 1: Get redirect URL
        base_url = f"https://source.unsplash.com/featured/?{query},{random.randint(1000,9999)}"
        response = requests.get(base_url, allow_redirects=False)
        if response.status_code == 302:
            redirected_url = response.headers['Location']
            # Step 2: Download actual image
            img_response = requests.get(redirected_url, timeout=10)
            if img_response.status_code == 200:
                return Image.open(BytesIO(img_response.content)).convert("RGB")
    except Exception as e:
        print(f"❌ Error fetching image for query '{query}': {e}")
    return None

# --- Download images ---
def download_unsplash(category, save_dir):
    print(f"🔽 Downloading Unsplash images for: {category}")
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    retries = 0
    while count < NUM_IMAGES_PER_CATEGORY and retries < MAX_RETRIES:
        img = fetch_unsplash_image(category)
        if img:
            img = img.resize(IMG_SIZE)
            filename = f"{category}_{count}.jpg"
            path = os.path.join(save_dir, filename)
            img.save(path)
            count += 1
            retries = 0
        else:
            retries += 1
            time.sleep(1)
    print(f"✅ {count} images saved to {save_dir}")

# --- Run downloads ---
for label, term in categories.items():
    save_dir = os.path.join(SAVE_ROOT, label)
    download_unsplash(term, save_dir)