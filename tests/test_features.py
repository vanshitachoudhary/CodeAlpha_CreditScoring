import numpy as np
import pandas as pd

from features import add_features


def sample(**kw):
    base = dict(duration=12, credit_amount=2400, installment_commitment=2, age=30)
    base.update(kw)
    return pd.DataFrame([base])


def test_monthly_payment():
    assert add_features(sample())["monthly_payment"].iloc[0] == 200


def test_zero_duration_does_not_crash():
    out = add_features(sample(duration=0))
    assert np.isfinite(out["monthly_payment"].iloc[0])


def test_flags():
    out = add_features(sample(duration=36, age=22))
    assert out["long_duration"].iloc[0] == 1 and out["young_borrower"].iloc[0] == 1


def test_input_not_mutated():
    df = sample()
    add_features(df)
    assert "monthly_payment" not in df.columns
