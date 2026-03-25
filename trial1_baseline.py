"""
TRIAL 1: BASELINE — NO RULES
Pure independent profit maximization. No memory, no learning, no constraints.
Each airline myopically picks the price that maximises THIS period's profit.
Expected: Bertrand competition → race to marginal cost.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# ── Parameters ──────────────────────────────────────────────
num_days = 1000
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
base_costs = np.ones(num_airlines) * 65
alpha = 0.08  # softer price sensitivity for logit
market_demand_per_day = 1000
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

print("[TRIAL 1] Baseline — No Rules (Independent Profit Maximisation)")
print(f"  Duration : {num_days} days")
print(f"  Airlines : {num_airlines}")
print(f"  Cost     : ${base_costs[0]}")
print()

data = {a: {'prices': [], 'profits': [], 'market_shares': []} for a in airlines}
prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])

for day in range(num_days):
    if day % 250 == 0:
        print(f"  Day {day}/{num_days}  avg price ${prices.mean():.1f}")

    # Logit demand
    exp_u = np.exp(-alpha * prices)
    shares = exp_u / exp_u.sum()
    demands = shares * market_demand_per_day
    profits = (prices - base_costs) * demands

    for i, a in enumerate(airlines):
        data[a]['prices'].append(prices[i])
        data[a]['profits'].append(profits[i])
        data[a]['market_shares'].append(shares[i] * 100)

    # Strategy: test a few price changes, keep best profit
    new_prices = np.zeros(num_airlines)
    for i in range(num_airlines):
        candidates = prices[i] * np.array([0.94, 0.97, 1.0, 1.03, 1.06])
        best_p, best_pi = prices[i], -np.inf
        for cp in candidates:
            tp = prices.copy(); tp[i] = cp
            eu = np.exp(-alpha * tp)
            sh = eu / eu.sum()
            pi = (cp - base_costs[i]) * sh[i] * market_demand_per_day
            if pi > best_pi:
                best_pi = pi; best_p = cp
        new_prices[i] = max(best_p, base_costs[i] + 0.01)
    prices = new_prices

# ── Plotting ────────────────────────────────────────────────
print("\n[TRIAL 1] Creating visualisation …")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=100)
fig.suptitle('Trial 1: Baseline — No Rules (Race to Marginal Cost)', fontsize=14, y=0.98)
days_range = range(num_days)

for i, a in enumerate(airlines):
    ax1.plot(days_range, data[a]['prices'], label=a, color=colors[i], linewidth=1.5)
ax1.axhline(y=65, color='black', ls=':', lw=1, alpha=.5, label='Cost ($65)')
ax1.set_ylabel("Price ($)")
ax1.set_title("Price Trajectories")
ax1.legend(loc='best', fontsize=8, ncol=6)
ax1.grid(True, alpha=.15)
ax1.set_xlim(0, num_days)

for i, a in enumerate(airlines):
    ax2.plot(days_range, data[a]['market_shares'], color=colors[i], linewidth=1.5)
ax2.set_ylabel("Market Share (%)")
ax2.set_xlabel("Day")
ax2.set_title("Market Share")
ax2.grid(True, alpha=.15)
ax2.set_xlim(0, num_days)
ax2.set_ylim(0, 60)

plt.tight_layout()
output_dir = "trials/trial1_baseline"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/trial1_300dpi.png", dpi=300, bbox_inches='tight', facecolor='white')

final = {"Airline": airlines,
         "Final_Price": [data[a]['prices'][-1] for a in airlines],
         "Avg_Price":   [np.mean(data[a]['prices']) for a in airlines]}
pd.DataFrame(final).to_csv(f"{output_dir}/final_stats.csv", index=False)
print(f"[TRIAL 1] Saved to {output_dir}/\n")
