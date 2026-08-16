import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/final_dataset.csv")
sns.set_style("whitegrid")

plt.figure(figsize=(10,6))
friction_counts = df[df['friction_type'] != 'other']['friction_type'].value_counts()
friction_counts.plot(kind='bar', color='steelblue')
plt.title("Digital Friction Types — Complaint Volume (excluding uncategorized)")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("data/processed/friction_distribution.png")
plt.close()

plt.figure(figsize=(10,6))
platform_sentiment = pd.crosstab(df['platform'], df['sentiment'], normalize='index') * 100
platform_sentiment.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title("Sentiment Breakdown by Platform")
plt.ylabel("% of Reviews")
plt.tight_layout()
plt.savefig("data/processed/platform_sentiment.png")
plt.close()

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