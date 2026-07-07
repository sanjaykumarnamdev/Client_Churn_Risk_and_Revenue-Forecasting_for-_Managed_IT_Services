"""
Collapse the client-month panel into ONE ROW PER CLIENT (the actual modeling
table), matching the brief: 200 clients, ~18% churn, 12% missing financial data.

For each client we pick a "decision point" month and compute trailing-3-month
aggregate features as of that point, with the label = did the client churn
within the following 3 months.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

panel = pd.read_csv("./client_month_panel.csv")
clients_static = pd.read_csv("./clients_static.csv")

rows = []
for cid, g in panel.groupby("client_id"):
    g = g.sort_values("month").reset_index(drop=True)
    max_month = g["month"].max()
    eventually_churned = bool(g["client_eventually_churned"].iloc[0])

    # decision point: for churned clients, pick a point 1-5 months before their last month
    # (so the 3-month-ahead label is well defined); for retained clients, pick any point
    # with at least 3 trailing months of history and >=3 months of runway in the panel.
    if eventually_churned and len(g) >= 4:
        offset = rng.integers(1, min(5, len(g)))
        decision_idx = len(g) - 1 - offset
        decision_idx = max(decision_idx, 2)
    else:
        if len(g) < 4:
            continue
        decision_idx = rng.integers(2, len(g) - 1)

    decision_row = g.iloc[decision_idx]
    trailing = g.iloc[max(0, decision_idx - 2): decision_idx + 1]  # trailing 3 months

    label = 1 if (eventually_churned and (max_month - decision_row["month"]) <= 3) else 0

    static = clients_static[clients_static.client_id == cid].iloc[0]

    rows.append({
        "client_id": cid,
        "industry": static.industry,
        "size_tier": static.size_tier,
        "n_services": static.n_services,
        "tenure_months": int(decision_row["month"] - static.tenure_start_month + 1),
        "avg_ticket_volume_3mo": round(trailing.ticket_volume.mean(), 2),
        "avg_sla_breaches_3mo": round(trailing.sla_breaches.mean(), 2),
        "max_sla_breach_streak_3mo": int(trailing.sla_breach_streak.max()),
        "avg_escalation_rate_3mo": round(trailing.escalation_rate.mean(), 3),
        "avg_csat_3mo": round(trailing.csat.mean(), 2),
        "avg_support_sentiment_3mo": round(trailing.support_sentiment.mean(), 3),
        "contact_tenure_months": int(decision_row.contact_tenure_months),
        "avg_discount_pct_3mo": round(trailing.discount_pct.mean(), 2),
        "avg_invoice_delay_days_3mo": round(trailing.invoice_delay_days.mean(), 2),
        "months_to_renewal": int(decision_row.months_to_renewal),
        "monthly_revenue": trailing.revenue.mean() if trailing.revenue.notna().any() else np.nan,
        "will_churn_3mo": label,
    })

client_df = pd.DataFrame(rows)

# introduce 12% missingness specifically on the financial field to match brief,
# on top of whatever came through from the panel's own missing-revenue months
extra_missing_mask = rng.random(len(client_df)) < 0.09
client_df.loc[extra_missing_mask, "monthly_revenue"] = np.nan

# ---------------------------------------------------------------------------
# Re-derive the churn label as an explicit function of the engineered features
# (rather than inheriting the raw monthly-hazard simulation label). The panel
# simulation above is stochastic month-by-month, which washes out signal by
# the time it's aggregated to one row per client. For a clean, learnable demo
# dataset, each client is scored on a business-sensible risk formula plus
# noise, then thresholded to match the brief's ~18% churn rate. Feature
# distributions (means, missingness, correlations) still come from the
# realistic panel simulation above -- only label generation is tightened so
# the downstream model has real signal to find, same as it would need on
# real client data of this size.
# ---------------------------------------------------------------------------
def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-9)

risk_score = (
    0.9 * zscore(client_df.avg_escalation_rate_3mo)
    + 0.8 * zscore(client_df.avg_sla_breaches_3mo)
    + 0.7 * zscore(client_df.max_sla_breach_streak_3mo)
    - 0.8 * zscore(client_df.avg_support_sentiment_3mo)
    - 0.6 * zscore(client_df.avg_csat_3mo)
    - 0.7 * zscore(client_df.contact_tenure_months)
    + 0.6 * zscore(client_df.avg_invoice_delay_days_3mo)
    - 0.5 * zscore(client_df.tenure_months)
    + rng.normal(0, 0.9, len(client_df))  # irreducible noise so it's not a trivial rule
)

churn_threshold = np.quantile(risk_score, 1 - 0.18)  # top 18% = churn
client_df["will_churn_3mo"] = (risk_score >= churn_threshold).astype(int)

print("Client-level dataset shape:", client_df.shape)
print("Churn rate:", client_df.will_churn_3mo.mean().round(3))
print("Missing revenue rate:", client_df.monthly_revenue.isna().mean().round(3))

client_df.to_csv("./client_modeling_dataset.csv", index=False)
print("Saved client_modeling_dataset.csv")
