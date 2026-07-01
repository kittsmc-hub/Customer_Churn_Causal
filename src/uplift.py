"""
Turn per-customer CATE into an actionable targeting policy with a real
ROI number, not just a model score.

Business inputs (override via CLI or import):
  discount_cost: $ cost of giving the coupon to one customer
  customer_value: $ monthly margin retained if a customer does NOT churn
  retention_horizon_months: how many months of margin a saved customer
    is worth (simple finite-horizon approximation, not full CLV)

Decision rule: offer the discount to customer i if and only if the
expected value of doing so is positive:
    expected_value_i = (-cate_i) * customer_value * retention_horizon_months
                        - discount_cost
(-cate_i because cate_i is the effect on churn probability; a negative
cate means the discount reduces churn, i.e. -cate_i is the retention
gain in probability terms.)

This is uplift modeling's actual job: stop spending discount budget on
customers who would not have churned anyway (lost causes / sleeping
dogs) and on customers a discount cannot save, and concentrate it on
persuadables where the math pencils out.

Run: python3 src/uplift.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

DEFAULT_DISCOUNT_COST = 15.0
DEFAULT_CUSTOMER_VALUE = 45.0  # avg monthly margin
DEFAULT_RETENTION_HORIZON_MONTHS = 6


@dataclass
class UpliftSegment:
    name: str
    n_customers: int
    pct_of_base: float
    mean_cate: float
    expected_value_per_customer: float
    total_expected_value: float
    recommend_discount: bool


@dataclass
class ROIReport:
    discount_cost: float
    customer_value: float
    retention_horizon_months: int
    naive_policy_value: float  # if you discount everyone
    no_action_value: float  # if you discount no one (always 0 by construction)
    targeted_policy_value: float  # discount only where expected_value > 0
    segments: list


def expected_value_per_customer(
    cate: pd.Series,
    discount_cost: float = DEFAULT_DISCOUNT_COST,
    customer_value: float = DEFAULT_CUSTOMER_VALUE,
    retention_horizon_months: int = DEFAULT_RETENTION_HORIZON_MONTHS,
) -> pd.Series:
    retention_gain_prob = -cate  # cate is effect on churn prob; flip sign for retention
    expected_retained_value = retention_gain_prob * customer_value * retention_horizon_months
    return expected_retained_value - discount_cost


def segment_customers(
    df: pd.DataFrame,
    cate_col: str = "cate",
    discount_cost: float = DEFAULT_DISCOUNT_COST,
    customer_value: float = DEFAULT_CUSTOMER_VALUE,
    retention_horizon_months: int = DEFAULT_RETENTION_HORIZON_MONTHS,
) -> tuple[pd.DataFrame, ROIReport]:
    df = df.copy()
    ev = expected_value_per_customer(df[cate_col], discount_cost, customer_value, retention_horizon_months)
    df["expected_value"] = ev
    df["recommend_discount"] = ev > 0

    # Four-quadrant uplift taxonomy, defined on the CATE itself
    # (cate is the effect on churn probability; more negative = bigger
    # retention win from discounting):
    persuasion_threshold = -0.08  # at least 8pp absolute churn reduction
    backfire_threshold = 0.0

    def classify(row_cate: float) -> str:
        if row_cate <= persuasion_threshold:
            return "persuadable"
        if row_cate > backfire_threshold:
            return "sleeping_dog"  # discount increases churn risk
        return "lost_cause"  # discount has negligible effect either way

    df["segment"] = df[cate_col].apply(classify)

    n_total = len(df)
    segments = []
    for seg_name in ["persuadable", "lost_cause", "sleeping_dog"]:
        seg_df = df[df["segment"] == seg_name]
        if len(seg_df) == 0:
            continue
        segments.append(
            UpliftSegment(
                name=seg_name,
                n_customers=len(seg_df),
                pct_of_base=len(seg_df) / n_total,
                mean_cate=float(seg_df[cate_col].mean()),
                expected_value_per_customer=float(seg_df["expected_value"].mean()),
                total_expected_value=float(seg_df.loc[seg_df["recommend_discount"], "expected_value"].sum()),
                recommend_discount=bool(seg_df["expected_value"].mean() > 0),
            )
        )

    naive_policy_value = float(ev.sum())  # discount everyone
    no_action_value = 0.0
    targeted_policy_value = float(ev[ev > 0].sum())  # discount only EV-positive customers

    report = ROIReport(
        discount_cost=discount_cost,
        customer_value=customer_value,
        retention_horizon_months=retention_horizon_months,
        naive_policy_value=naive_policy_value,
        no_action_value=no_action_value,
        targeted_policy_value=targeted_policy_value,
        segments=[asdict(s) for s in segments],
    )
    return df, report


if __name__ == "__main__":
    df = pd.read_csv("outputs/cate_estimates.csv")
    df_segmented, report = segment_customers(df)

    print(json.dumps(asdict(report), indent=2))

    uplift_vs_naive = report.targeted_policy_value - report.naive_policy_value
    pct_budget_saved = 1 - (df_segmented["recommend_discount"].sum() / len(df_segmented))
    print(
        f"\nTargeting only EV-positive customers beats discounting everyone "
        f"by ${uplift_vs_naive:,.0f} in expected value across this {len(df_segmented):,}-customer "
        f"base, while withholding the discount from "
        f"{pct_budget_saved:.1%} of customers it would have wasted budget on."
    )

    df_segmented.to_csv("outputs/uplift_segments.csv", index=False)
    print("\nWrote outputs/uplift_segments.csv")
