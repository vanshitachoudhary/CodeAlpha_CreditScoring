"""Feature engineering - train.py aur app.py dono yahi use karte hain,
taaki training aur prediction mein same features bane."""
import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Monthly bojh: loan amount ko duration (months) se baanta
    df["monthly_payment"] = df["credit_amount"] / df["duration"].clip(lower=1)
    # Loan amount bahut skewed hota hai, log se normal ke kareeb aata hai
    df["log_credit_amount"] = np.log1p(df["credit_amount"])
    # Installment ka income pe asar (installment_commitment = % of disposable income)
    df["burden_score"] = df["installment_commitment"] * df["monthly_payment"]
    # Lamba loan = zyada risk
    df["long_duration"] = (df["duration"] > 24).astype(int)
    # Bahut young borrowers
    df["young_borrower"] = (df["age"] < 25).astype(int)
    return df
