# Indian Digital Life Friction Dataset

A data analytics project that measures **where Indians face friction while using digital services** — UPI payments, e-commerce apps, and government digital platforms — by mining real Google Play Store reviews.

> Millions of Indians write 1-star reviews about payment failures, KYC delays, and app crashes every day — that's free, real, unstructured data about a real problem. This project turns that into a structured dataset and finds patterns in it.

---

## Key findings

*(from ~4,400 Play Store reviews across 7 apps: PhonePe, Google Pay, Paytm, Amazon, Flipkart, DigiLocker, mAadhaar)*

- **Amazon had the worst negative sentiment at 51.7%** — more than double every other platform (next worst: Flipkart at 31.6%)
- **Payment apps performed best** — Paytm (16.7%) and PhonePe (18.0%) had the lowest negative sentiment of all 7 apps
- **The *type* of friction differs by category.** E-commerce apps (Amazon, Flipkart) are dominated by `customer_support` and `delivery_issue` complaints — service failures. Government apps (DigiLocker, mAadhaar) are dominated by `performance_issue` and `app_crash` — engineering/stability failures. Same "friction" label, different root cause.
- **Most common friction type overall:** `customer_support` (297), `performance_issue` (155), `app_crash` (99), `delivery_issue` (68), `fraud_or_scam` (65)

**Why this is a strong portfolio project:**
- It's an original dataset — not a Kaggle CSV everyone has already used
- Covers the full pipeline: collection → cleaning → labeling → sentiment analysis → SQL → visualization
- Has a real "so what" — insights a product/fintech team could actually act on

---

## Project structure

```
digital-friction-project/
├── data/
│   ├── raw/                        # untouched scraped data
│   │   └── playstore_reviews.csv
│   ├── processed/                  # cleaned, labeled, final outputs
│   │   ├── combined_clean.csv
│   │   ├── labeled_data.csv
│   │   ├── final_dataset.csv
│   │   ├── friction.db
│   │   ├── friction_distribution.png
│   │   ├── platform_sentiment.png
│   │   └── monthly_trend.png
├── scripts/
│   ├── scrape_playstore.py         # step 1: collect data
│   ├── clean_data.py               # step 2: clean text
│   ├── label_friction.py           # step 3: tag friction type
│   ├── sentiment_analysis.py       # step 4: VADER sentiment
│   ├── load_to_sql.py              # step 5: load into SQLite
│   └── visualize.py                # step 6: charts
├── dashboard/
│   └── app.py                      # optional Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

**Interviewer might ask:** *"Why scripts instead of one notebook?"*
**Answer:** Notebooks are great for exploration, but scripts are modular and reusable — you can re-run `scrape_playstore.py` on a schedule without re-running the whole analysis. It also mirrors how production pipelines are actually organized.

---

## Setup

```bash
git clone <your-repo-url>
cd digital-friction-project
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 1 — Collect data from Google Play Store reviews

No API key needed — `google-play-scraper` is free and open.

```python
# scripts/scrape_playstore.py
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
    result, _ = reviews(
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
```

**Why pull ALL ratings, not just 1-3 star?**
If you only scrape negative reviews, sentiment analysis becomes meaningless — you've already filtered for negative sentiment before analyzing it. Pull everything and let the analysis show what's negative.

**Why not Reddit/Twitter too?**
Both were tried during this project. Twitter's API is now paid. Consumer-complaint websites like consumercomplaints.in render their content with JavaScript, so a simple `requests` call returns an empty shell — nothing for BeautifulSoup to parse (you'd need Selenium/Playwright to get around that, which is a fair v2 addition but overkill for v1). Play Store alone gave a large, clean, real dataset without any of that overhead.

---

## Step 2 — Clean the data

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
    combined = pd.read_csv("data/raw/playstore_reviews.csv")
    combined['clean_text'] = combined['text'].apply(clean_text)
    combined = combined[combined['clean_text'].str.len() > 10]
    combined.drop_duplicates(subset='clean_text', inplace=True)
    combined.to_csv("data/processed/combined_clean.csv", index=False)
    print(f"Final dataset: {len(combined)} rows")
    return combined

if __name__ == "__main__":
    load_and_combine()
```

---

## Step 3 — Label the friction type (the core NLP step)

A rule-based keyword classifier — built and refined by inspecting real "uncategorized" text and iterating on the keyword list until the categories captured the actual complaint patterns in the data.

```python
# scripts/label_friction.py
import pandas as pd

FRICTION_KEYWORDS = {
    "payment_failure": [
        "payment failed", "transaction failed", "money deducted", "amount debited",
        "payment not done", "payment issue", "payment problem", "money gone",
        "amount deducted", "transaction unsuccessful", "mandate not approved",
        "mandate issue", "mandate rejected"
    ],
    "otp_issue": [
        "otp not received", "otp delay", "otp expired", "wrong otp",
        "otp issue", "otp problem", "no otp"
    ],
    "refund_delay": [
        "refund not received", "refund pending", "waiting for refund",
        "money not refunded", "refund issue", "refund problem", "no refund"
    ],
    "app_crash": [
        "app crash", "app not working", "app hangs", "force close",
        "app crashing", "stopped working", "not working properly", "app freeze",
        "stuck", "loading screen", "not opening", "won't open", "can not open",
        "unable to open"
    ],
    "kyc_issue": [
        "kyc pending", "kyc failed", "kyc rejected", "verification failed",
        "kyc issue", "verification pending", "kyc problem"
    ],
    "customer_support": [
        "customer care", "customer support", "no response", "support useless",
        "complaint ignored", "toll free", "no cooperation", "not responding",
        "poor service", "no proper response", "customer service", "rude",
        "no proper solution"
    ],
    "login_issue": [
        "cannot login", "login failed", "account locked", "cannot access account",
        "unable to login", "sign in problem", "login issue"
    ],
    "delivery_issue": [
        "late delivery", "delivery delay", "not delivered", "delivery problem",
        "delayed delivery", "order late", "delivery issue", "delivery agent",
        "delivery boy", "not delivered on time", "shipment delay"
    ],
    "compatibility_issue": [
        "not compatible", "unable to download", "cannot install", "device not supported",
        "not installing", "compatibility issue"
    ],
    "balance_discrepancy": [
        "wrong balance", "improper balance", "balance mismatch", "incorrect balance",
        "showing wrong", "balance issue", "glitch"
    ],
    "document_issue": [
        "document fetching", "document not found", "unable to fetch document",
        "document error", "document problem", "fetching problem"
    ],
    "performance_issue": [
        "slow app", "very slow", "app is slow", "lagging", "hangs a lot",
        "bakwas", "bakwash", "worst app", "useless app", "bekar"
    ],
    "damaged_or_wrong_product": [
        "damaged product", "damaged package", "different item", "wrong item",
        "wrong product", "broken product", "defective product"
    ],
    "fraud_or_scam": [
        "scam", "fraud", "fraudulent", "fraudulently"
    ],
    "order_cancelled": [
        "order cancelled", "automatically cancelled", "cancelled my order",
        "order got cancelled"
    ],
    "security_issue": [
        "security threat", "security issue", "obfuscation", "malware", "unsafe app"
    ],
}

def assign_friction_type(text):
    text = str(text).lower()
    for friction_type, keywords in FRICTION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return friction_type
    return "other"

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
**Answer:** "For v1, I wanted something explainable and fast to build with zero labeled training data. A rule-based approach is 100% interpretable — I know exactly why something got labeled `payment_failure`. Once I have this labeled dataset, I could train a supervised classifier using these rule-based labels as a starting point — a real technique called weak supervision. That would be my v2 improvement."

**Interviewer might ask:** *"Why does 'other' still make up most of the dataset?"*
**Answer:** "Two reasons. First, a large share of the reviews are genuinely positive — 'best app', 'awesome' — and correctly have no friction type. Second, the remaining negative-but-uncategorized text is a long tail of one-off complaints that don't repeat often enough to justify a dedicated category. I checked this directly: I filtered 'other' rows down to just the negative-sentiment ones and read a sample before deciding which new categories were worth adding — that's how `damaged_or_wrong_product`, `fraud_or_scam`, `order_cancelled`, and `security_issue` were identified."

---

## Step 4 — Sentiment analysis

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
**Answer:** VADER is lexicon-based and tuned specifically for short, informal text like reviews — exactly what this dataset is. It needs no GPU or training and is fully explainable. BERT would be more accurate but is overkill for v1 and much harder to explain if asked how it works internally. Picking the right-sized tool is the point.

---

## Step 5 — Load into SQL and query it

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

Queries used to produce the key findings above:

```sql
-- Top friction type per platform
SELECT platform, friction_type, COUNT(*) as complaint_count
FROM complaints
WHERE friction_type != 'other'
GROUP BY platform, friction_type
ORDER BY platform, complaint_count DESC;

-- Platform with worst negative sentiment %
SELECT platform,
       ROUND(SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS negative_pct,
       COUNT(*) AS total_reviews
FROM complaints
GROUP BY platform
ORDER BY negative_pct DESC;

-- Most common friction type overall
SELECT friction_type, COUNT(*) as cnt
FROM complaints
WHERE friction_type != 'other'
GROUP BY friction_type
ORDER BY cnt DESC;
```

**Interviewer might ask:** *"Why SQLite and not Postgres/MySQL?"*
**Answer:** SQLite needs zero setup — it's a file, not a server — perfect for a portfolio project people can clone and run instantly. In production I'd use Postgres for concurrent access, but the SQL logic itself (joins, aggregations, grouping) transfers directly.

---

## Step 6 — Visualization

```python
# scripts/visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/final_dataset.csv")
sns.set_style("whitegrid")

# 1. Friction type distribution — 'other' excluded deliberately, see note below
plt.figure(figsize=(10,6))
friction_counts = df[df['friction_type'] != 'other']['friction_type'].value_counts()
friction_counts.plot(kind='bar', color='steelblue')
plt.title("Digital Friction Types — Complaint Volume (excluding uncategorized)")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("data/processed/friction_distribution.png")
plt.close()

# 2. Sentiment breakdown by platform
plt.figure(figsize=(10,6))
platform_sentiment = pd.crosstab(df['platform'], df['sentiment'], normalize='index') * 100
platform_sentiment.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title("Sentiment Breakdown by Platform")
plt.ylabel("% of Reviews")
plt.tight_layout()
plt.savefig("data/processed/platform_sentiment.png")
plt.close()

# 3. Monthly trend
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')
monthly = df.groupby('month').size()
plt.figure(figsize=(10,6))
monthly.plot(kind='line', marker='o')
plt.title("Complaint Volume Over Time")
plt.tight_layout()
plt.savefig("data/processed/monthly_trend.png")
plt.close()

print("Charts saved to data/processed/")
```

**Why exclude "other" from the friction-type chart?**
"Other" is ~80% of the dataset by volume (mostly positive reviews with no friction to categorize). Including it in the same bar chart as the real categories would visually flatten everything else into invisible slivers. Excluding it makes the actual pattern — what's breaking, and how often — visible.

---

## Step 7 (bonus) — Streamlit dashboard

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
    fig1 = px.bar(filtered[filtered['friction_type'] != 'other']['friction_type'].value_counts(),
                   title="Friction Types")
    st.plotly_chart(fig1)
with col2:
    fig2 = px.pie(filtered, names='sentiment', title="Sentiment Split")
    st.plotly_chart(fig2)

st.dataframe(filtered[['platform', 'friction_type', 'sentiment', 'text']].head(50))
```

Run with: `streamlit run dashboard/app.py`


## Final checklist before publishing to GitHub

- [x] `requirements.txt` lists exact packages used
- [x] `.gitignore` excludes `venv/`, raw/processed CSVs, and the SQLite `.db` file
- [x] README has real findings and real numbers, not placeholders
- [x] Top of README has a scannable summary (recruiters skim the first 10 seconds)
- [ ] Add 1-2 actual chart screenshots into this README (embed the PNGs from `data/processed/`)
- [ ] Write a LinkedIn post summarizing the top 2-3 findings, linking to the repo
