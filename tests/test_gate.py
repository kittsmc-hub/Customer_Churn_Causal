"""
Gate tests: deterministic, local, free, must run in well under 2s total.
No model fitting here -- that's covered by the eval suite in
tests/eval_causal_quality.py, which is allowed to be slow/non-deterministic
because it trains models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.generate_data import generate
from src.uplift import expected_value_per_customer, segment_customers


# ---------- data generation ----------

def test_generate_is_deterministic():
    df1 = generate(n=500, seed=7)
    df2 = generate(n=500, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_different_seeds_differ():
    df1 = generate(n=500, seed=1)
    df2 = generate(n=500, seed=2)
    assert not df1["churned"].equals(df2["churned"])


def test_generate_schema_and_ranges():
    df = generate(n=2000, seed=42)
    expected_cols = {
        "customer_id", "tenure_months", "contract", "internet_service",
        "monthly_charges", "payment_method", "support_calls",
        "has_dependents", "senior_citizen", "paperless_billing",
        "discount_offered", "churned", "_true_effect_logit", "_true_persuadable",
    }
    assert expected_cols.issubset(set(df.columns))
    assert df["customer_id"].is_unique
    assert df["tenure_months"].between(0, 72).all()
    assert df["monthly_charges"].between(15, 150).all()
    assert df["churned"].isin([0, 1]).all()
    assert df["discount_offered"].isin([0, 1]).all()
    assert set(df["contract"].unique()) <= {"month-to-month", "one-year", "two-year"}


def test_treatment_is_confounded_not_random():
    """The whole point of this dataset is confounded treatment assignment.
    If treatment rate doesn't vary with risk factors, the demo is testing
    nothing -- catch that regression here."""
    df = generate(n=5000, seed=42)
    month_to_month_rate = df.loc[df.contract == "month-to-month", "discount_offered"].mean()
    two_year_rate = df.loc[df.contract == "two-year", "discount_offered"].mean()
    assert month_to_month_rate != pytest.approx(two_year_rate, abs=0.02)


def test_no_missing_values():
    df = generate(n=1000, seed=3)
    assert df.drop(columns=["_true_effect_logit", "_true_persuadable"]).isna().sum().sum() == 0


# ---------- uplift / ROI math ----------

def test_expected_value_formula_sign():
    # A CATE of -1.0 (huge churn reduction) at high customer value should
    # always be EV-positive; a CATE of +1.0 (discount backfires) should
    # always be EV-negative, regardless of cost.
    cate = pd.Series([-1.0, 1.0])
    ev = expected_value_per_customer(cate, discount_cost=15.0, customer_value=45.0, retention_horizon_months=6)
    assert ev.iloc[0] > 0
    assert ev.iloc[1] < 0


def test_expected_value_zero_cate_is_just_negative_cost():
    cate = pd.Series([0.0])
    ev = expected_value_per_customer(cate, discount_cost=15.0, customer_value=45.0, retention_horizon_months=6)
    assert ev.iloc[0] == pytest.approx(-15.0)


def test_expected_value_scales_linearly_with_horizon():
    cate = pd.Series([-0.1])
    ev_3mo = expected_value_per_customer(cate, retention_horizon_months=3)
    ev_6mo = expected_value_per_customer(cate, retention_horizon_months=6)
    # cost is horizon-independent, so the *gain* component should double,
    # not the whole EV -- check via the gain component directly.
    gain_3mo = ev_3mo.iloc[0] + 15.0
    gain_6mo = ev_6mo.iloc[0] + 15.0
    assert gain_6mo == pytest.approx(2 * gain_3mo)


def test_segment_customers_buckets_are_exhaustive_and_disjoint():
    df = pd.DataFrame({"cate": [-0.3, -0.05, 0.05, -0.09]})
    out, report = segment_customers(df)
    assert set(out["segment"].unique()) <= {"persuadable", "lost_cause", "sleeping_dog"}
    assert len(out) == 4
    assert out["segment"].notna().all()


def test_segment_customers_persuadable_threshold():
    df = pd.DataFrame({"cate": [-0.081, -0.079]})
    out, _ = segment_customers(df)
    assert out.loc[out.cate == -0.081, "segment"].iloc[0] == "persuadable"
    assert out.loc[out.cate == -0.079, "segment"].iloc[0] != "persuadable"


def test_targeted_policy_never_worse_than_naive_policy_in_expectation():
    """Targeting (discount only EV-positive customers) must, by
    construction, never produce a lower total expected value than
    discounting everyone -- it's a strict superset restriction to the
    positive-EV subset. Regression test for the EV-sum logic itself."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cate": rng.uniform(-0.3, 0.3, size=2000)})
    _, report = segment_customers(df)
    assert report.targeted_policy_value >= report.naive_policy_value


def test_segment_customers_handles_all_one_segment():
    df = pd.DataFrame({"cate": [-0.5, -0.4, -0.3]})  # all persuadable
    out, report = segment_customers(df)
    assert (out["segment"] == "persuadable").all()
    assert len(report.segments) == 1
    assert report.segments[0]["name"] == "persuadable"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
