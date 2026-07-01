"""
Generate a synthetic telecom churn dataset with an embedded intervention
(discount coupon) and a known, heterogeneous causal effect on churn.

Why synthetic instead of raw Telco-Customer-Churn: the public Kaggle/IBM
dataset has no treatment column and no recorded randomization, so there is
no ground truth to validate a CATE estimator against. We keep its feature
distributions (tenure, monthly charges, contract type, support calls) but
add a treatment assignment mechanism with deliberate confounding (sales
reps target high-risk, high-tenure customers preferentially) and a known
per-customer treatment effect. This lets the eval suite check the causal
estimator against ground truth, which is impossible with real-world data
where the true effect is never observed.

Ground-truth effect design (so the eval can grade itself):
  - Customers with month-to-month contracts AND tenure < 12 are highly
    persuadable: discount cuts their churn probability by ~18pp.
  - Customers on long contracts (one/two year) are near-zero uplift:
    they were not going to churn anyway, discount wastes margin.
  - Customers with >=4 support calls are net-negative responders: the
    discount reads as desperation and slightly increases churn (-3pp
    effect on retention, i.e. +3pp churn), modeling the real-world
    "coupon signals the relationship is failing" effect.
This is the persuadables-vs-sure-things-vs-lost-causes-vs-sleeping-dogs
structure that uplift modeling exists to disentangle.

This script is deterministic: same seed -> byte-identical CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 10_000

CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
INTERNET_TYPES = ["dsl", "fiber", "none"]
PAYMENT_TYPES = ["electronic-check", "mailed-check", "bank-transfer", "credit-card"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = N_CUSTOMERS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.gamma(shape=2.0, scale=18.0, size=n).clip(0, 72).round().astype(int)
    contract = rng.choice(CONTRACT_TYPES, size=n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET_TYPES, size=n, p=[0.35, 0.45, 0.20])
    monthly_charges = np.where(
        internet == "fiber",
        rng.normal(85, 15, n),
        np.where(internet == "dsl", rng.normal(55, 12, n), rng.normal(25, 8, n)),
    ).clip(15, 150)
    payment_method = rng.choice(PAYMENT_TYPES, size=n, p=[0.30, 0.20, 0.25, 0.25])
    support_calls = rng.poisson(lam=1.3, size=n).clip(0, 12)
    has_dependents = rng.binomial(1, 0.30, size=n)
    senior_citizen = rng.binomial(1, 0.16, size=n)
    paperless_billing = rng.binomial(1, 0.59, size=n)

    contract_risk = np.select(
        [contract == "month-to-month", contract == "one-year", contract == "two-year"],
        [1.0, 0.35, 0.10],
    )

    # Baseline (no-treatment) churn risk score in logit space.
    base_logit = (
        -1.4
        + 1.8 * contract_risk
        + 0.018 * monthly_charges
        - 0.035 * tenure_months
        + 0.22 * support_calls
        - 0.30 * has_dependents
        + 0.15 * (payment_method == "electronic-check")
        + 0.10 * senior_citizen
        - 0.05 * paperless_billing
    )

    # --- Treatment assignment (confounded, NOT random) ---
    # Sales/retention reps preferentially offer discounts to customers who
    # already look high-risk and to higher-tenure customers (retention
    # budget policy). This creates classic confounding: treated customers
    # are disproportionately high-churn-risk, so a naive "treated vs
    # untreated churn rate" comparison is biased toward making the
    # discount look useless or harmful even where it truly helps.
    propensity_logit = (
        -0.8
        + 1.1 * _sigmoid(base_logit)  # rep targets high baseline risk
        + 0.01 * tenure_months  # rep also favors higher-tenure accounts
        + 0.25 * (contract == "month-to-month")
    )
    propensity = _sigmoid(propensity_logit).clip(0.05, 0.95)
    treatment = rng.binomial(1, propensity)

    # --- True heterogeneous treatment effect (ground truth, hidden from
    # any estimator; used only to generate outcomes and to grade evals) ---
    persuadable = (contract == "month-to-month") & (tenure_months < 12)
    lost_cause_long_contract = contract != "month-to-month"
    sleeping_dog = support_calls >= 4

    true_effect_logit = np.where(
        persuadable,
        -1.45,  # ~18pp absolute churn reduction near the operating point
        np.where(
            sleeping_dog,
            0.28,  # discount backfires: +effect on churn logit
            np.where(lost_cause_long_contract, -0.05, -0.55),  # near-zero / moderate
        ),
    )

    outcome_logit = base_logit + treatment * true_effect_logit
    churn_prob = _sigmoid(outcome_logit)
    churned = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{idx:05d}" for idx in range(n)],
            "tenure_months": tenure_months,
            "contract": contract,
            "internet_service": internet,
            "monthly_charges": monthly_charges.round(2),
            "payment_method": payment_method,
            "support_calls": support_calls,
            "has_dependents": has_dependents,
            "senior_citizen": senior_citizen,
            "paperless_billing": paperless_billing,
            "discount_offered": treatment,
            "churned": churned,
            # Hidden ground truth, kept ONLY for eval grading. The modeling
            # code must never read this column as a feature.
            "_true_effect_logit": true_effect_logit.round(4),
            "_true_persuadable": persuadable.astype(int),
        }
    )
    return df


if __name__ == "__main__":
    out = generate()
    out.to_csv("data/churn_data.csv", index=False)
    print(f"Wrote {len(out)} rows to data/churn_data.csv")
    print(f"Overall churn rate: {out['churned'].mean():.3f}")
    print(f"Treatment rate: {out['discount_offered'].mean():.3f}")
    print(
        "Naive (confounded) churn rate, treated vs untreated: "
        f"{out.loc[out.discount_offered == 1, 'churned'].mean():.3f} vs "
        f"{out.loc[out.discount_offered == 0, 'churned'].mean():.3f}"
    )
