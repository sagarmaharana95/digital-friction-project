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