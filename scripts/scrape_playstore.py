from google_play_scraper import reviews, Sort
import pandas as pd
import time

APPS = {
    "PhonePe": "com.phonepe.app",
    "Google Pay": "com.google.android.apps.nbu.paisa.user",
    "Paytm": "net.one97.paytm",
    "Amazon": "in.amazon.mShop.android.shopping",
    "Flipkart": "com.flipkart.android",
    "DigiLocker": "com.digilocker.android",
    "mAadhaar": "in.gov.uidai.mAadhaarPlus",
}

CATEGORY_MAP = {
    "PhonePe": "payments", "Google Pay": "payments", "Paytm": "payments",
    "Amazon": "ecommerce", "Flipkart": "ecommerce",
    "DigiLocker": "govt", "mAadhaar": "govt",
}

def scrape_app(app_name, package_name, count=1500):
    all_reviews = []
    result, continuation_token = reviews(
        package_name, lang='en', country='in',
        sort=Sort.NEWEST, count=count, filter_score_with=None
    )
    for r in result:
        all_reviews.append({
            "platform": app_name, "category": CATEGORY_MAP[app_name],
            "text": r['content'], "rating": r['score'], "date": r['at'],
            "thumbs_up": r['thumbsUpCount'], "source": "play_store"
        })
    return all_reviews

if __name__ == "__main__":
    everything = []
    for name, pkg in APPS.items():
        print(f"Scraping {name}...")
        try:
            everything.extend(scrape_app(name, pkg))
        except Exception as e:
            print(f"Failed on {name}: {e}")
        time.sleep(2)

    df = pd.DataFrame(everything)
    df.to_csv("data/raw/playstore_reviews.csv", index=False)
    print(f"Saved {len(df)} reviews")
