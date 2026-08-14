# Indian Digital Life Friction Dataset — Full Project Guide

A complete data analytics project that measures **where Indians face friction while using digital services** — UPI payments, e-commerce apps, and government digital platforms. This guide walks through every step with code, and explains *why* each step exists so you can defend it in an interview.

---

## 0. The idea, in one line

> "Millions of Indians write 1-star reviews and Reddit rants about payment failures, KYC delays, and app crashes every day — that's free, real, unstructured data about a real problem. This project turns that into a structured dataset and finds patterns in it."

**Why this is a good portfolio project (say this in interviews):**
- It's *your own* dataset — not a Kaggle CSV everyone has used
- It touches the full pipeline: collection → cleaning → labeling → NLP → SQL → visualization
- It has a real-world "so what" — insights a fintech/product team could actually use

---

## 1. Project setup

```bash
mkdir digital-friction-project
cd digital-friction-project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install google-play-scraper praw pandas numpy nltk vaderSentiment matplotlib seaborn plotly sqlite3 streamlit
```

Folder structure (keep it clean — recruiters open GitHub repos and judge by structure):

```
digital-friction-project/
├── data/
│   ├── raw/              # untouched scraped data
│   └── processed/        # cleaned, labeled data
├── scripts/
│   ├── scrape_playstore.py
│   ├── scrape_reddit.py
│   ├── clean_data.py
│   ├── label_friction.py
│   ├── sentiment_analysis.py
│   └── visualize.py
├── notebooks/
│   └── eda.ipynb
├── dashboard/
│   └── app.py
├── README.md
└── requirements.txt
```

**Interviewer might ask:** *"Why not just do everything in one notebook?"*
**Answer:** Notebooks are great for exploration, but scripts are modular and reusable — you can re-run `scrape_playstore.py` weekly without re-running your whole analysis. Shows you think about production-style organization, not just one-off analysis.

---

## 2. Step 1 — Collect data from Google Play Store reviews

This is your primary data source. `google-play-scraper` needs no API key.

```python
# scripts/scrape_playstore.py
from google_play_scraper import reviews, Sort
import pandas as pd
import time

# App package names (found in Play Store URL after id=)
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
        package_name,
        lang='en',
        country='in',
        sort=Sort.NEWEST,
        count=count,
        filter_score_with=None   # we want all ratings, not just low ones
    )
    for r in result:
        all_reviews.append({
            "platform": app_name,
            "category": CATEGORY_MAP[app_name],
            "text": r['content'],
            "rating": r['score'],
            "date": r['at'],
            "thumbs_up": r['thumbsUpCount'],
            "source": "play_store"
        })
    return all_reviews

if __name__ == "__main__":
    everything = []
    for name, pkg in APPS.items():
        print(f"Scraping {name}...")
        everything.extend(scrape_app(name, pkg))
        time.sleep(2)   # be polite, avoid rate limiting

    df = pd.DataFrame(everything)
    df.to_csv("data/raw/playstore_reviews.csv", index=False)
    print(f"Saved {len(df)} reviews")
```

**Why pull ALL ratings, not just 1-3 star?**
Interviewer trap question. Answer: if you only scrape negative reviews, your sentiment analysis is meaningless (you've already filtered for negative sentiment before analyzing it — that's data leakage/bias). Pull everything, then *let the analysis show* what's negative.

**Interviewer might ask:** *"What if the scraper breaks / Play Store changes structure?"*
**Answer:** Wrap in try/except per app so one failure doesn't kill the whole run; log failed apps and retry separately. (Add this in your real script — shows resilience thinking.)

---

## 3. Step 2 — Collect data from Reddit

```python
# scripts/scrape_reddit.py
import praw
import pandas as pd

# Get these free from https://www.reddit.com/prefs/apps (create a "script" app)
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
                    "platform": "reddit_general",
                    "category": "unlabeled",     # will assign in cleaning step
                    "text": post.title + " " + (post.selftext or ""),
                    "rating": None,
                    "date": pd.to_datetime(post.created_utc, unit='s'),
                    "thumbs_up": post.score,
                    "source": "reddit",
                    "search_keyword": keyword
                })
    return pd.DataFrame(all_posts)

if __name__ == "__main__":
    df = scrape_reddit_complaints()
    df.drop_duplicates(subset="text", inplace=True)
    df.to_csv("data/raw/reddit_posts.csv", index=False)
    print(f"Saved {len(df)} Reddit posts")
```

**Interviewer might ask:** *"Why Reddit and not Twitter/X?"*
**Answer:** Twitter's API is paid now (as of the API changes), Reddit's is free and has an official Python wrapper (PRAW). Reddit also has more longform, contextual complaints — Twitter is short and noisy.

---

## 4. Step 3 — Clean and combine the data

```python
# scripts/clean_data.py
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
```

**Interviewer might ask:** *"Why regex and not just an NLP library for cleaning?"*
**Answer:** Regex is faster and fully explainable for basic cleaning (URLs, special chars). NLP libraries (spaCy/NLTK) come in later for the smarter stuff — tokenization, stopwords, lemmatization — which happens right before sentiment analysis, not during raw cleaning.

---

## 5. Step 4 — Label the "friction type" (this is the core NLP/analytics step)

This is what makes it *your* dataset, not just scraped raw text.

```python
# scripts/label_friction.py
import pandas as pd

# Rule-based keyword classifier — simple, explainable, good enough for v1
FRICTION_KEYWORDS = {
    "payment_failure": ["payment failed", "transaction failed", "money deducted", "amount debited"],
    "otp_issue": ["otp not received", "otp delay", "otp expired", "wrong otp"],
    "refund_delay": ["refund not received", "refund pending", "waiting for refund", "money not refunded"],
    "app_crash": ["app crash", "app not working", "app hangs", "force close"],
    "kyc_issue": ["kyc pending", "kyc failed", "kyc rejected", "verification failed"],
    "customer_support": ["customer care", "no response", "support useless", "complaint ignored"],
    "login_issue": ["cannot login", "login failed", "account locked", "cannot access account"],
}

def assign_friction_type(text):
    text = str(text).lower()
    matches = []
    for friction_type, keywords in FRICTION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            matches.append(friction_type)
    if not matches:
        return "other"
    return matches[0]   # take first match; a text could have multiple, keep it simple for v1

def label_dataset():
    df = pd.read_csv("data/processed/combined_clean.csv")
    df['friction_type'] = df['clean_text'].apply(assign_friction_type)
    df.to_csv("data/processed/labeled_data.csv", index=False)
    print(df['friction_type'].value_counts())
    return df

if __name__ == "__main__":
    label_dataset()
```

**Interviewer might ask:** *"Why rule-based keywords and not a trained ML classifier?"*
**Answer (this is an important answer — memorize the logic):**
"For v1, I wanted something explainable and fast to build with zero labeled training data. A rule-based approach is 100% interpretable — I know exactly why something got labeled `payment_failure`. Once I have this labeled dataset, I could train a supervised classifier (like Naive Bayes or a fine-tuned small model) using these rule-based labels as a starting point — that's actually a `real technique called weak supervision`. That would be my v2 improvement."

This answer shows you know the limitation AND the next step — exactly what interviewers want to hear.

---

## 6. Step 5 — Sentiment Analysis

```python
# scripts/sentiment_analysis.py
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = analyzer.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

def add_sentiment():
    df = pd.read_csv("data/processed/labeled_data.csv")
    df['sentiment'] = df['clean_text'].apply(get_sentiment)
    df.to_csv("data/processed/final_dataset.csv", index=False)
    print(df['sentiment'].value_counts())
    return df

if __name__ == "__main__":
    add_sentiment()
```

**Interviewer might ask:** *"Why VADER and not a transformer model like BERT?"*
**Answer:** VADER is rule-based, lexicon-driven, and specifically tuned for short, informal text like reviews and social posts — which is exactly what this dataset is. It's fast, needs no GPU/training, and is fully explainable. A BERT-based model would be more accurate but is overkill for a v1 project and much harder to explain in an interview if asked "how does it work internally." Good engineering is picking the right-sized tool, not the fanciest one.

---

## 7. Step 6 — Load into SQL and query it (shows you're not just a pandas person)

```python
# scripts/load_to_sql.py
import pandas as pd
import sqlite3

df = pd.read_csv("data/processed/final_dataset.csv")
conn = sqlite3.connect("data/friction.db")
df.to_sql("complaints", conn, if_exists="replace", index=False)
conn.close()
print("Loaded into SQLite: data/friction.db")
```

Now run real SQL analysis — this is what you screenshot for your portfolio/README:

```sql
-- Which friction type is most common per platform?
SELECT platform, friction_type, COUNT(*) as complaint_count
FROM complaints
GROUP BY platform, friction_type
ORDER BY platform, complaint_count DESC;

-- Which platform has the worst average sentiment?
SELECT platform,
       SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS negative_pct
FROM complaints
GROUP BY platform
ORDER BY negative_pct DESC;

-- Monthly trend of payment_failure complaints (window function — shows SQL depth)
SELECT strftime('%Y-%m', date) AS month,
       COUNT(*) AS failures,
       SUM(COUNT(*)) OVER (ORDER BY strftime('%Y-%m', date)) AS running_total
FROM complaints
WHERE friction_type = 'payment_failure'
GROUP BY month
ORDER BY month;
```

**Interviewer might ask:** *"Why SQLite and not Postgres/MySQL?"*
**Answer:** For a project of this size, SQLite needs zero setup (it's a file, not a server) — perfect for a portfolio project people can clone and run instantly. In a production environment, I'd use Postgres for concurrent access and scalability, but the SQL logic itself (joins, window functions, aggregations) transfers directly.

---

## 8. Step 7 — Visualization

```python
# scripts/visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/final_dataset.csv")
sns.set_style("whitegrid")

# 1. Friction type distribution
plt.figure(figsize=(10,6))
df['friction_type'].value_counts().plot(kind='bar', color='steelblue')
plt.title("Digital Friction Types — Complaint Volume")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data/processed/friction_distribution.png")

# 2. Platform comparison — % negative sentiment
platform_sentiment = pd.crosstab(df['platform'], df['sentiment'], normalize='index') * 100
platform_sentiment.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title("Sentiment Breakdown by Platform")
plt.ylabel("% of Reviews")
plt.tight_layout()
plt.savefig("data/processed/platform_sentiment.png")

# 3. Monthly trend
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')
monthly = df.groupby('month').size()
plt.figure(figsize=(10,6))
monthly.plot(kind='line', marker='o')
plt.title("Complaint Volume Over Time")
plt.tight_layout()
plt.savefig("data/processed/monthly_trend.png")

print("Charts saved to data/processed/")
```

---

## 9. Step 8 (bonus) — Simple Streamlit dashboard

```python
# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Indian Digital Life Friction Dashboard")

df = pd.read_csv("data/processed/final_dataset.csv")

platform_filter = st.multiselect("Select platforms", df['platform'].unique(), default=df['platform'].unique())
filtered = df[df['platform'].isin(platform_filter)]

col1, col2 = st.columns(2)
with col1:
    fig1 = px.bar(filtered['friction_type'].value_counts(), title="Friction Types")
    st.plotly_chart(fig1)
with col2:
    fig2 = px.pie(filtered, names='sentiment', title="Sentiment Split")
    st.plotly_chart(fig2)

st.dataframe(filtered[['platform', 'friction_type', 'sentiment', 'text']].head(50))
```

Run with: `streamlit run dashboard/app.py`

---

## 10. Interview Q&A cheat-sheet (read this before interviews)

**Q: Walk me through your project end to end.**
A: "I scraped Play Store reviews and Reddit posts about Indian digital services — payments, e-commerce, govt apps. I cleaned the text, built a rule-based classifier to tag each complaint with a friction type like payment failure or KYC issue, ran sentiment analysis with VADER, loaded everything into SQLite for querying, and visualized the patterns — which platform has the worst friction, what type of issue is most common, and how it trends over time."

**Q: What was the hardest part?**
A: "Labeling friction type without any pre-existing labeled data. I solved it with a rule-based keyword classifier for v1, which is explainable, and noted that a supervised model trained on those labels would be the natural v2 improvement."

**Q: What would you improve with more time?**
A: "Three things: (1) train an actual ML classifier instead of rule-based friction labeling, (2) add Twitter/X data if I had API budget, (3) build a proper Postgres backend instead of SQLite for a live-updating dashboard."

**Q: What insight surprised you?** *(Answer this with YOUR actual numbers once you run it — interviewers can tell when this is memorized vs real)*

**Q: How is this different from a Kaggle project?**
A: "Kaggle datasets are already clean and labeled — someone did the hard part for you. Here I did the full pipeline myself: collection, cleaning, and labeling, which is what actual data analyst work looks like."

---

## 11. Final checklist before publishing to GitHub

- [ ] `requirements.txt` with all packages and versions
- [ ] `.gitignore` excludes `venv/`, API keys, large raw CSVs if too big
- [ ] README has your actual charts/screenshots embedded (not just this guide)
- [ ] Top of README has a 3-line summary + key finding (recruiters skim first 10 seconds)
- [ ] Reddit API keys stored in `.env`, never committed
- [ ] LinkedIn post written summarizing your top 2-3 findings, linking to repo

Good luck Dz — is level ka execution kaafi students se aage rakhega tumhe. Ab bas isko run karo, apna real data collect karo, aur apne actual numbers ke saath Q&A section fill karo.
