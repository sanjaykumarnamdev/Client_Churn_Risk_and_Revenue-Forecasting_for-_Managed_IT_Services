"""
12-month revenue forecast, churn-adjusted via Monte Carlo simulation.

For each client:
  - base monthly revenue = current monthly_revenue (imputed if missing)
  - monthly retention probability derived from their 3-month churn probability
    (convert to an implied monthly hazard, compound forward)
  - simulate N scenarios of client survival month by month, sum revenue

Also produces a "reduce churn by 5 points" scenario for comparison.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)

scored = pd.read_csv("./client_risk_scores.csv")

N_SIM = 2000
N_MONTHS = 12

# monthly hazard implied by the 3-month churn probability: p3 = 1-(1-h)^3 -> h = 1-(1-p3)^(1/3)
p3 = scored["churn_probability"].clip(0.001, 0.95).values
monthly_hazard = 1 - (1 - p3) ** (1 / 3)
revenue = scored["monthly_revenue"].fillna(scored["monthly_revenue"].median()).values
n_clients = len(scored)

# slight organic monthly growth assumption for retained clients (uplift/inflation)
growth_rate = 0.003


def simulate(hazard_array, n_sim=N_SIM):
    totals = np.zeros(n_sim)
    monthly_totals = np.zeros((n_sim, N_MONTHS))
    for s in range(n_sim):
        alive = np.ones(n_clients, dtype=bool)
        for m in range(N_MONTHS):
            # clients churn this month according to their hazard, only if still alive
            churn_draw = rng.random(n_clients) < hazard_array
            alive = alive & (~churn_draw) if m > 0 else (~churn_draw)
            month_rev = np.where(alive, revenue * (1 + growth_rate) ** m, 0).sum()
            monthly_totals[s, m] = month_rev
        totals[s] = monthly_totals[s].sum()
    return totals, monthly_totals


# --- Scenario A: business as usual ---
totals_base, monthly_base = simulate(monthly_hazard)

# --- Scenario B: retention strategy cuts churn by 5 points (absolute, on 3-month prob) ---
p3_reduced = np.clip(p3 - 0.05, 0.001, 0.95)
monthly_hazard_reduced = 1 - (1 - p3_reduced) ** (1 / 3)
totals_reduced, monthly_reduced = simulate(monthly_hazard_reduced)

print("=== 12-Month Revenue Forecast (Monte Carlo, N=%d) ===" % N_SIM)
print(f"\nScenario A - Business as usual:")
print(f"  Expected revenue: ${totals_base.mean():,.0f}")
print(f"  90% CI: ${np.percentile(totals_base, 5):,.0f} - ${np.percentile(totals_base, 95):,.0f}")

print(f"\nScenario B - Churn reduced by 5 points (retention strategy):")
print(f"  Expected revenue: ${totals_reduced.mean():,.0f}")
print(f"  90% CI: ${np.percentile(totals_reduced, 5):,.0f} - ${np.percentile(totals_reduced, 95):,.0f}")

uplift = totals_reduced.mean() - totals_base.mean()
print(f"\nRevenue uplift from 5-point churn reduction: ${uplift:,.0f} ({uplift/totals_base.mean()*100:.1f}%)")

# --- monthly trajectory plot ---
months = np.arange(1, N_MONTHS + 1)
base_mean = monthly_base.mean(axis=0)
base_p5 = np.percentile(monthly_base, 5, axis=0)
base_p95 = np.percentile(monthly_base, 95, axis=0)
red_mean = monthly_reduced.mean(axis=0)

plt.figure(figsize=(9, 5.5))
plt.fill_between(months, base_p5, base_p95, color="#4C6EF5", alpha=0.15, label="90% CI (as usual)")
plt.plot(months, base_mean, color="#4C6EF5", lw=2.5, label="Business as usual")
plt.plot(months, red_mean, color="#2F9E44", lw=2.5, ls="--", label="Churn -5pts (retention strategy)")
plt.xlabel("Month")
plt.ylabel("Monthly Revenue ($)")
plt.title("12-Month Revenue Forecast: Business-as-usual vs. Retention Strategy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./revenue_forecast.png", dpi=150)
plt.close()

# --- distribution plot ---
plt.figure(figsize=(8, 5))
plt.hist(totals_base, bins=40, alpha=0.6, color="#4C6EF5", label="Business as usual")
plt.hist(totals_reduced, bins=40, alpha=0.6, color="#2F9E44", label="Churn -5pts")
plt.xlabel("Total 12-Month Revenue ($)")
plt.ylabel("Simulations")
plt.title("Revenue Distribution: 2,000 Monte Carlo Simulations")
plt.legend()
plt.tight_layout()
plt.savefig("./revenue_distribution.png", dpi=150)
plt.close()

# save summary
with open("./revenue_forecast_summary.txt", "w") as f:
    f.write("12-Month Revenue Forecast Summary\n")
    f.write("=" * 40 + "\n")
    f.write(f"Scenario A (business as usual): ${totals_base.mean():,.0f}  "
            f"[90% CI ${np.percentile(totals_base,5):,.0f} - ${np.percentile(totals_base,95):,.0f}]\n")
    f.write(f"Scenario B (churn -5pts): ${totals_reduced.mean():,.0f}  "
            f"[90% CI ${np.percentile(totals_reduced,5):,.0f} - ${np.percentile(totals_reduced,95):,.0f}]\n")
    f.write(f"Revenue uplift: ${uplift:,.0f} ({uplift/totals_base.mean()*100:.1f}%)\n")
    f.write(f"Revenue-at-risk from churn (12mo): ${revenue.sum()*12 - totals_base.mean():,.0f} "
            f"vs. a no-churn baseline\n")

print("\nSaved: revenue_forecast.png, revenue_distribution.png, revenue_forecast_summary.txt")
