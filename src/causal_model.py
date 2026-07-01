"""
Causal effect of the discount coupon on churn, estimated with DoWhy
(identification + refutation) and EconML's LinearDML / CausalForestDML
(conditional average treatment effect, CATE).

Question: does offering customer i a discount change P(churn_i)? Not "is
discount_offered predictive of churn" -- that's the baseline model's
question and it answers a different thing.

Causal graph (encoded explicitly, not inferred):
  confounders (tenure, contract, monthly_charges, support_calls,
  has_dependents, senior_citizen, paperless_billing, internet_service,
  payment_method) -> discount_offered (treatment)
  confounders -> churned (outcome)
  discount_offered -> churned (the effect we want)

This matches the data-generating process in data/generate_data.py: reps
assign discounts based on observed risk signals (confounded treatment),
and those same signals independently drive churn.

Run: python3 src/causal_model.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from dowhy import CausalModel
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import HistGradientBoostingClassifier as GradientBoostingClassifier, HistGradientBoostingRegressor as GradientBoostingRegressor
from sklearn.model_selection import train_test_split

CONFOUNDERS = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "has_dependents",
    "senior_citizen",
    "paperless_billing",
]
CATEGORICAL_CONFOUNDERS = ["contract", "internet_service", "payment_method"]
TREATMENT = "discount_offered"
OUTCOME = "churned"


@dataclass
class ATEResult:
    ate: float
    ate_stderr: float
    ate_ci_lower: float
    ate_ci_upper: float


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[CONFOUNDERS].copy()
    dummies = pd.get_dummies(df[CATEGORICAL_CONFOUNDERS], drop_first=True)
    return pd.concat([X, dummies], axis=1)


def build_causal_model(df: pd.DataFrame) -> CausalModel:
    """Build the DoWhy CausalModel with an explicit DAG (no auto-discovery:
    the structure is a modeling assumption that must be stated, not
    inferred from a heuristic)."""
    confounder_str = " ".join(CONFOUNDERS + CATEGORICAL_CONFOUNDERS)
    gml = (
        "graph[directed 1 "
        f'node[id "{TREATMENT}" label "{TREATMENT}"] '
        f'node[id "{OUTCOME}" label "{OUTCOME}"] '
        + " ".join(f'node[id "{c}" label "{c}"]' for c in CONFOUNDERS + CATEGORICAL_CONFOUNDERS)
        + f' edge[source "{TREATMENT}" target "{OUTCOME}"] '
        + " ".join(f'edge[source "{c}" target "{TREATMENT}"]' for c in CONFOUNDERS + CATEGORICAL_CONFOUNDERS)
        + " ".join(f'edge[source "{c}" target "{OUTCOME}"]' for c in CONFOUNDERS + CATEGORICAL_CONFOUNDERS)
        + "]"
    )
    model = CausalModel(
        data=df,
        treatment=TREATMENT,
        outcome=OUTCOME,
        graph=gml,
    )
    return model


def identify_and_estimate_ate(df: pd.DataFrame, seed: int = 42) -> tuple[CausalModel, ATEResult, object]:
    """Identification is done through DoWhy against the explicit DAG
    (documents the assumption, checks the backdoor criterion). Estimation
    is done by calling EconML's LinearDML directly: DoWhy's estimate_effect
    wrapper recomputes bootstrap confidence intervals by refitting the
    full nuisance models dozens of times, which is needless when LinearDML
    already provides fast analytic (HC0/asymptotic) inference."""
    model = build_causal_model(df)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    X = _design_matrix(df)
    T = df[TREATMENT].values
    Y = df[OUTCOME].values

    estimator = LinearDML(
        model_y=GradientBoostingRegressor(random_state=seed),
        model_t=GradientBoostingClassifier(random_state=seed),
        discrete_treatment=True,
        random_state=seed,
        cv=3,
    )
    estimator.fit(Y, T, X=None, W=X)

    ate = float(estimator.ate(X=None))
    ate_lower, ate_upper = (float(v) for v in estimator.ate_interval(X=None))
    inference = estimator.ate_inference(X=None)
    result = ATEResult(
        ate=ate,
        ate_stderr=float(inference.stderr_mean),
        ate_ci_lower=ate_lower,
        ate_ci_upper=ate_upper,
    )
    return model, result, estimator, identified_estimand


def _fit_ate(df: pd.DataFrame, seed: int) -> float:
    X = _design_matrix(df)
    T = df[TREATMENT].values
    Y = df[OUTCOME].values
    estimator = LinearDML(
        model_y=GradientBoostingRegressor(random_state=seed),
        model_t=GradientBoostingClassifier(random_state=seed),
        discrete_treatment=True,
        random_state=seed,
        cv=3,
    )
    estimator.fit(Y, T, X=None, W=X)
    return float(estimator.ate(X=None))


def run_refutation_tests(df: pd.DataFrame, original_ate: float, seed: int = 42, num_simulations: int = 8) -> dict:
    """Three refutation tests, each designed to break a fragile estimate.
    Implemented as direct refits (not DoWhy's bootstrap wrapper) so each
    test costs ~1-2s instead of triggering nested bootstrap CIs.

    placebo_treatment: shuffle the treatment column so it carries no real
        signal about the outcome. A robust estimator should report an
        effect close to 0 here; if the original estimate also survives a
        shuffled treatment, the original estimate was not really using the
        treatment column and should not be trusted.
    data_subset: refit on random 80% subsamples. A real, identifiable
        effect should be stable in sign and rough magnitude across subsets;
        wild swings indicate the estimate is unstable / underpowered.
    random_common_cause: add a feature drawn from pure noise into the
        confounder set. Since it is independent of everything, adding it
        should not move the ATE estimate; if it does, the estimator is
        overfitting to spurious structure.
    """
    rng = np.random.default_rng(seed)
    results: dict = {}

    placebo_effects = []
    for i in range(num_simulations):
        shuffled = df.copy()
        shuffled[TREATMENT] = rng.permutation(shuffled[TREATMENT].values)
        placebo_effects.append(_fit_ate(shuffled, seed=seed + i))
    results["placebo_treatment"] = {
        "expectation": "effect should collapse toward 0 when treatment is randomly permuted",
        "original_ate": original_ate,
        "placebo_ate_mean": float(np.mean(placebo_effects)),
        "placebo_ate_std": float(np.std(placebo_effects)),
        "passes": bool(abs(np.mean(placebo_effects)) < abs(original_ate) * 0.5),
    }

    subset_effects = []
    for i in range(num_simulations):
        subset = df.sample(frac=0.8, random_state=seed + i)
        subset_effects.append(_fit_ate(subset, seed=seed + i))
    results["data_subset"] = {
        "expectation": "effect should be stable in sign/magnitude on random 80% subsets",
        "original_ate": original_ate,
        "subset_ate_mean": float(np.mean(subset_effects)),
        "subset_ate_std": float(np.std(subset_effects)),
        "passes": bool(np.sign(np.mean(subset_effects)) == np.sign(original_ate)
                        and np.std(subset_effects) < abs(original_ate)),
    }

    noise_effects = []
    for i in range(num_simulations):
        noisy = df.copy()
        noisy["_random_noise_confounder"] = rng.normal(size=len(noisy))
        X = _design_matrix(noisy)
        X["_random_noise_confounder"] = noisy["_random_noise_confounder"].values
        T = noisy[TREATMENT].values
        Y = noisy[OUTCOME].values
        estimator = LinearDML(
            model_y=GradientBoostingRegressor(random_state=seed + i),
            model_t=GradientBoostingClassifier(random_state=seed + i),
            discrete_treatment=True,
            random_state=seed + i,
            cv=3,
        )
        estimator.fit(Y, T, X=None, W=X)
        noise_effects.append(float(estimator.ate(X=None)))
    results["random_common_cause"] = {
        "expectation": "adding an independent random confounder should not move the estimate",
        "original_ate": original_ate,
        "noisy_ate_mean": float(np.mean(noise_effects)),
        "noisy_ate_std": float(np.std(noise_effects)),
        "passes": bool(abs(np.mean(noise_effects) - original_ate) < 0.5 * abs(original_ate) + 0.01),
    }

    return results


def estimate_cate(df: pd.DataFrame, seed: int = 42) -> tuple[CausalForestDML, pd.Series]:
    """Conditional Average Treatment Effect via Causal Forest DML: how
    much does the discount move churn probability for THIS customer,
    given their features. This is the number uplift targeting runs on."""
    X = _design_matrix(df)
    T = df[TREATMENT].values
    Y = df[OUTCOME].values

    est = CausalForestDML(
        model_y=GradientBoostingRegressor(random_state=seed),
        model_t=GradientBoostingClassifier(random_state=seed),
        discrete_treatment=True,
        n_estimators=400,
        min_samples_leaf=20,
        max_depth=None,
        random_state=seed,
    )
    est.fit(Y, T, X=X, W=None)
    cate = est.effect(X)
    return est, pd.Series(cate, index=df.index, name="cate")


if __name__ == "__main__":
    df = pd.read_csv("data/churn_data.csv")

    print("=== Step 1: Identification + ATE (LinearDML) ===")
    model, ate_result, estimator, identified_estimand = identify_and_estimate_ate(df)
    print(json.dumps(asdict(ate_result), indent=2))
    print(
        f"\nInterpretation: offering the discount changes churn probability "
        f"by {ate_result.ate:+.4f} on average across the population "
        f"(95% CI [{ate_result.ate_ci_lower:+.4f}, {ate_result.ate_ci_upper:+.4f}])."
    )
    print(f"\nIdentified estimand (backdoor adjustment set):\n{identified_estimand}")

    print("\n=== Step 2: Refutation tests ===")
    refutations = run_refutation_tests(df, ate_result.ate)
    print(json.dumps(refutations, indent=2))

    print("\n=== Step 3: CATE via CausalForestDML ===")
    est, cate = estimate_cate(df)
    df_out = df.copy()
    df_out["cate"] = cate
    print(df_out[["customer_id", "contract", "tenure_months", "support_calls", "cate"]].describe())
    df_out.to_csv("outputs/cate_estimates.csv", index=False)
    print("\nWrote outputs/cate_estimates.csv")
