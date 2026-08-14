import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)          # remove URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)      # remove special chars/emoji
    text = re.sub(r'\s+', ' ', text).strip()      # collapse whitespace
    return text

def load_and_combine():
    play = pd.read_csv("data/raw/playstore_reviews.csv")
    reddit = pd.read_csv("data/raw/reddit_posts.csv")

    combined = pd.concat([play, reddit], ignore_index=True)
    combined['clean_text'] = combined['text'].apply(clean_text)

    # Drop empty/too-short entries (not useful for analysis)
    combined = combined[combined['clean_text'].str.len() > 10]

    # Drop exact duplicates (bots / repeated spam reviews)
    combined.drop_duplicates(subset='clean_text', inplace=True)

    combined.to_csv("data/processed/combined_clean.csv", index=False)
    print(f"Final dataset: {len(combined)} rows")
    return combined

if __name__ == "__main__":
    load_and_combine()