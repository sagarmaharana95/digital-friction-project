import pandas as pd

# Expanded, broader keyword sets based on real review text patterns.
# Each category uses shorter/more flexible phrases so more real complaints match.
FRICTION_KEYWORDS = {
    "payment_failure": [
        "payment failed", "transaction failed", "money deducted", "amount debited",
        "payment not done", "payment issue", "payment problem", "money gone",
        "amount deducted", "transaction unsuccessful", "mandate not approved",
        "mandate issue", "mandate rejected", "paisa pay nhi"
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
        "not installing", "compatibility issue", "not compatible with this version"
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
        "order got cancelled", "orders are getting automatically cancelled"
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