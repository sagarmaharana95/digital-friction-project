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