"""
Generate synthetic 'transaction level' data for 200 enterprise IT clients
and aggregate into a client-month panel. This simulates the 10M-row raw
transaction dataset described in the brief (12% missing financial data,
~18% overall churn rate).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_CLIENTS = 200
N_MONTHS = 36  # 3 years of history
SERVICES = ["infra_mgmt", "cloud_migration", "security", "support_247"]

# ---------- 1. Client static attributes ----------
industries = ["Finance", "Healthcare", "Retail", "Manufacturing", "Tech", "Government", "Education"]
client_ids = [f"CL{str(i).zfill(4)}" for i in range(1, N_CLIENTS + 1)]

clients = pd.DataFrame({
    "client_id": client_ids,
    "industry": rng.choice(industries, N_CLIENTS),
    "size_tier": rng.choice(["Small", "Mid", "Large"], N_CLIENTS, p=[0.4, 0.4, 0.2]),
    "n_services": rng.integers(1, 5, N_CLIENTS),
    "base_monthly_revenue": rng.gamma(shape=4, scale=3000, size=N_CLIENTS) + 2000,
    "tenure_start_month": rng.integers(0, 12, N_CLIENTS),  # some joined partway through window
})

# latent "risk profile" per client drives churn probability later
clients["latent_risk"] = rng.beta(2, 5, N_CLIENTS)  # skewed toward lower risk, some high-risk tail

# ---------- 2. Build client-month panel ----------
rows = []
for _, c in clients.iterrows():
    active = True
    churned_month = None
    sla_breach_streak = 0
    contact_tenure = rng.integers(3, 24)
    cum_revenue_trend = 0

    for m in range(N_MONTHS):
        if m < c.tenure_start_month:
            continue
        if not active:
            break

        # monthly evolving features
        usage_trend = rng.normal(0, 1) - 0.05 * cum_revenue_trend * c.latent_risk
        ticket_volume = max(0, rng.poisson(8 + 20 * c.latent_risk))
        sla_breaches = rng.poisson(0.3 + 2 * c.latent_risk)
        sla_breach_streak = sla_breach_streak + 1 if sla_breaches > 1 else 0
        escalation_rate = np.clip(rng.normal(0.1 + 0.3 * c.latent_risk, 0.05), 0, 1)
        csat = np.clip(rng.normal(4.2 - 1.5 * c.latent_risk, 0.4), 1, 5)
        sentiment_score = np.clip(rng.normal(0.3 - 0.6 * c.latent_risk, 0.2), -1, 1)
        contact_turnover_event = rng.random() < (0.02 + 0.05 * c.latent_risk)
        if contact_turnover_event:
            contact_tenure = 0
        else:
            contact_tenure += 1
        discount_pct = np.clip(rng.normal(5 + 10 * c.latent_risk, 3), 0, 40)
        invoice_delay_days = max(0, rng.normal(3 + 15 * c.latent_risk, 5))
        months_to_renewal = 12 - (m % 12)

        revenue = c.base_monthly_revenue * (1 + 0.01 * usage_trend) * (1 - discount_pct / 100)
        cum_revenue_trend += usage_trend

        # 12% chance a financial field is missing this month (systematic-ish: high risk clients slightly more likely)
        financial_missing = rng.random() < (0.10 + 0.05 * c.latent_risk)

        # ---- churn hazard model (drives the label) ----
        hazard = (
            0.0015
            + 0.014 * c.latent_risk
            + 0.005 * sla_breach_streak
            + 0.009 * escalation_rate
            + 0.005 * max(0, -sentiment_score)
            + 0.004 * (contact_tenure < 2)
            + 0.004 * (invoice_delay_days > 20)
            + 0.005 * (months_to_renewal <= 1)
        )
        hazard = np.clip(hazard, 0, 0.30)

        rows.append({
            "client_id": c.client_id,
            "month": m,
            "industry": c.industry,
            "size_tier": c.size_tier,
            "n_services": c.n_services,
            "ticket_volume": ticket_volume,
            "sla_breaches": sla_breaches,
            "sla_breach_streak": sla_breach_streak,
            "escalation_rate": round(escalation_rate, 3),
            "csat": round(csat, 2),
            "support_sentiment": round(sentiment_score, 3),
            "contact_tenure_months": contact_tenure,
            "discount_pct": round(discount_pct, 2),
            "invoice_delay_days": round(invoice_delay_days, 1),
            "months_to_renewal": months_to_renewal,
            "revenue": None if financial_missing else round(revenue, 2),
            "financial_missing": financial_missing,
            "monthly_churn_hazard": round(hazard, 4),
        })

        if rng.random() < hazard:
            active = False
            churned_month = m

    # backfill churn label (will_churn_in_3mo) computed after loop in pandas

panel = pd.DataFrame(rows)

# ---------- 3. Label: will this client churn within the next 3 months? ----------
panel = panel.sort_values(["client_id", "month"]).reset_index(drop=True)
last_month_per_client = panel.groupby("client_id")["month"].transform("max")
n_months_per_client = panel.groupby("client_id")["month"].transform("count")
overall_last_month = panel["month"].max()

# a client "churned" (left before end of observation window) if their last observed month < overall last month
panel["client_left_window_end"] = panel.groupby("client_id")["month"].transform("max")
churned_clients = panel.groupby("client_id")["month"].max()
did_churn = (churned_clients < (N_MONTHS - 1))
panel["client_eventually_churned"] = panel["client_id"].map(did_churn)

# label = 1 if client's churn happens within 3 months of this row (and client does churn)
client_last_month = panel.groupby("client_id")["month"].transform("max")
panel["will_churn_3mo"] = np.where(
    panel["client_eventually_churned"],
    ((client_last_month - panel["month"]) <= 3).astype(int),
    0,
)

print("Panel shape:", panel.shape)
print("Overall client churn rate:", did_churn.mean().round(3))
print("Row-level positive label rate (3mo churn flag):", panel["will_churn_3mo"].mean().round(3))
print("Missing financial data rate:", panel["financial_missing"].mean().round(3))

panel.to_csv("./client_month_panel.csv", index=False)
clients.to_csv("./clients_static.csv", index=False)
print("Saved client_month_panel.csv and clients_static.csv")
