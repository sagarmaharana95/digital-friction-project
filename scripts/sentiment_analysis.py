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