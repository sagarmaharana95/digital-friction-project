import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_combine():
    combined = pd.read_csv("data/raw/playstore_reviews.csv")
    combined['clean_text'] = combined['text'].apply(clean_text)
    combined = combined[combined['clean_text'].str.len() > 10]
    combined.drop_duplicates(subset='clean_text', inplace=True)
    combined.to_csv("data/processed/combined_clean.csv", index=False)
    print(f"Final dataset: {len(combined)} rows")
    return combined

if __name__ == "__main__":
    load_and_combine()