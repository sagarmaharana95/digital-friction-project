import praw
import pandas as pd

reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="digital_friction_research"
)

SUBREDDITS = ["india", "developersIndia", "IndiaInvestments", "IndianStreetBets"]
KEYWORDS = [
    "UPI failed", "payment stuck", "KYC pending", "refund not received",
    "app crashed", "OTP not received", "DigiLocker error", "Aadhaar update failed"
]

def scrape_reddit_complaints(limit_per_query=100):
    all_posts = []
    for sub in SUBREDDITS:
        subreddit = reddit.subreddit(sub)
        for keyword in KEYWORDS:
            for post in subreddit.search(keyword, limit=limit_per_query):
                all_posts.append({
                    "platform": "reddit_general", "category": "unlabeled",
                    "text": post.title + " " + (post.selftext or ""),
                    "rating": None,
                    "date": pd.to_datetime(post.created_utc, unit='s'),
                    "thumbs_up": post.score, "source": "reddit",
                    "search_keyword": keyword
                })
    return pd.DataFrame(all_posts)

if __name__ == "__main__":
    df = scrape_reddit_complaints()
    df.drop_duplicates(subset="text", inplace=True)
    df.to_csv("data/raw/reddit_posts.csv", index=False)
    print(f"Saved {len(df)} Reddit posts")
