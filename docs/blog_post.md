# Why your churn model doesn't actually help retention

Your churn model has an AUC of 0.78. The team is happy. Marketing pulls the
top decile, sends everyone a discount, and three months later retention
hasn't moved. What happened?

The model was never built to answer the question you used it for.

## 1. The prediction vs causation trap

A churn classifier answers: *given everything I know about this customer,
including the fact that they already got a discount, how likely are they
to leave?* That is a prediction question, and it's a perfectly well-posed
one. But the business question is different: *if I offer this specific
customer a discount, will they be less likely to leave than if I don't?*
That's a causal question, and a predictive model cannot answer it, no
matter how high its AUC climbs.

Here's the trap concretely. Suppose your retention team already, sensibly,
gives discounts to customers who look high-risk -- the rep sees a
frustrated month-to-month customer and offers a coupon. Train a classifier
on that data and "received a discount" becomes one of the strongest
predictors of churn, because the customers who got discounts were already
the ones most likely to leave. Hand that model to marketing and they'll
read it backwards: "discounts predict churn, so discounts must not work
(or might even cause it)." The model isn't wrong. The question being asked
of it is.

On a synthetic telecom dataset built to have exactly this structure (reps
preferentially discount high-risk accounts, and the discount has a real
but customer-dependent effect on retention), a naive treated-vs-untreated
comparison undersells the discount's true impact by roughly 3.5 percentage
points. The correlation is biased in the misleading direction: it makes a
genuinely useful intervention look weaker than it is.

A baseline XGBoost classifier on this data reaches an AUC around 0.75 --
solidly predictive, completely useless for deciding who to discount.
Predictive accuracy and causal validity are orthogonal: you can max out
one while learning nothing true about the other.

## 2. DoWhy primer: DAGs, assumptions, and refutation tests

Causal inference from observational data (no randomized experiment) always
rests on an assumption you cannot fully verify from the data alone: that
you've measured every variable that affects both who gets treated and what
the outcome is. DoWhy's contribution is making that assumption explicit
instead of letting it hide inside a model's coefficients.

You start by drawing a DAG: which variables cause which. In a churn
setting, the confounders (tenure, contract type, support call volume,
monthly charges) cause both the treatment (whether a rep offers a
discount) and the outcome (whether the customer churns). The treatment
also independently causes the outcome -- that's the arrow you're trying to
size.

```
confounders --> treatment (discount offered)
confounders --> outcome (churned)
treatment --> outcome
```

Given that graph, DoWhy applies the backdoor criterion to derive the
adjustment set: control for every confounder, and the remaining
association between treatment and outcome is the causal effect. The
estimation itself is handled by a double machine learning estimator
(EconML's LinearDML): it residualizes both treatment and outcome on the
confounders using flexible models, then regresses the residuals on each
other. This orthogonalization is what makes the estimate robust to
confounders entering the data nonlinearly, which a naive linear regression
adjustment would miss.

The part most tutorials skip: identification and estimation are not the
end of the process. DoWhy's refutation tests exist specifically to try to
break your estimate before you trust it.

- **Placebo treatment**: randomly shuffle the treatment column, so it
  carries no real information, and re-estimate. If the "effect" survives a
  shuffled treatment, your original estimate wasn't using the treatment
  column at all -- it was picking up something else.
- **Data subset**: refit on random subsamples. A real effect should be
  stable in sign and rough magnitude; wild swings mean the estimate is
  underpowered or fragile.
- **Random common cause**: add a feature that's pure noise to the
  confounder set. A well-specified estimator shouldn't move much; if it
  does, something is overfitting.

On the dataset above, all three pass: the placebo effect collapses from
-8.3 points to roughly zero, the subsample estimates stay within a point
of each other, and adding noise barely shifts anything. That's the
evidence that lets you say "the discount reduces churn probability by 8.3
points on average" with more confidence than a regression coefficient
alone would justify.

## 3. CATE estimation walkthrough -- what the output actually means

The average treatment effect (ATE) answers "does the discount work, on
average, across everyone?" That's still not the number a retention team
needs, because "on average" hides enormous variation. The conditional
average treatment effect (CATE) answers "does the discount work for *this*
customer, given their specific situation?"

EconML's CausalForestDML estimates a CATE per customer by combining the
same double-ML orthogonalization with a forest that splits on which
features actually predict *heterogeneity* in the treatment effect -- not
which features predict the outcome, which is what a normal predictive
model's feature importance gives you.

On the synthetic data, the spread is the entire story: estimated CATE
ranges from roughly -29 percentage points (the discount sharply cuts churn
risk) down to +2 points (the discount very slightly raises churn risk for
a small segment). A single ATE of -8.3 points was always an average over
that whole range -- useful for a yes/no "does this intervention work"
decision, useless for a "who should get it" decision.

What does a CATE estimate of -22 points actually mean in practice? For
that one customer, the model's counterfactual claim is: holding their
tenure, contract, support history, and billing details fixed, offering
them a discount lowers their predicted churn probability by 22 points
relative to not offering one. It's not a prediction of what will happen --
it's an estimate of the *difference* two parallel-universe versions of
that customer would experience.

## 4. Uplift modeling: targeting the persuadables, not the sure things

Four conceptual buckets fall out of any treatment-effect distribution, and
they map directly onto a budget decision:

- **Persuadables**: the discount meaningfully reduces their churn risk.
  These are the only customers where spending money changes the outcome.
- **Sure things**: they were never going to churn regardless. A discount
  here is pure margin given away for free.
- **Lost causes**: they're leaving regardless of what you do. A discount
  here is also wasted, just for a different reason.
- **Sleeping dogs**: the intervention backfires. In this dataset, customers
  with four or more recent support calls respond to a discount coupon as a
  signal that the relationship is already failing, and their churn risk
  ticks up slightly when offered one.

A churn classifier's risk score cannot distinguish a persuadable from a
lost cause -- both look high-risk. Only the treatment-effect estimate can,
because it's measuring the *delta* the intervention produces, not the raw
risk level.

Turning CATE into a targeting policy just requires unit economics:

```
expected_value(customer) = (-CATE) × customer_value × retention_horizon − discount_cost
offer the discount  <=>  expected_value(customer) > 0
```

On the reference dataset (10,000 customers, $15 discount cost, $45/month
margin, six-month retention horizon), targeting only expected-value-positive
customers produces about $18,700 more expected value than blanket
discounting everyone, while withholding the discount from roughly a third
of the customer base it would have wasted budget on. None of that number
comes from a higher AUC. It comes entirely from asking a causal question
instead of a predictive one.

## 5. How causal ML thinking makes you a better data scientist generally

The habit this builds generalizes well past churn. Any time a model's
output is going to be used to decide an *intervention* -- who to call, who
to discount, who to flag for review, which feature to ship to which
segment -- the predictive-accuracy question and the causal-effect question
are different questions, and conflating them produces confidently wrong
decisions. A risk score tells you where to look. It does not tell you what
to do about it, and it especially does not tell you for whom doing
something will backfire.

The discipline DoWhy and EconML enforce -- draw the assumption explicitly
as a graph, name the confounders you're claiming are sufficient, and try
to break your own estimate before shipping it -- is good practice even
when you end up not using a formal causal framework. It's the difference
between a model that correlates and a model you can act on.
