"""
TRIAL 2: PRICE FLOOR + OBSERVATION
Adds two realistic features: a regulatory price floor ($90) and the ability
to observe competitors' average price from the previous period.
Airlines play myopic best-response against observed competitor average.
Expected: Prices converge to floor. Observation without learning ≠ coordination.
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
price_floor = 90.0  # regulatory minimum
alpha = 0.08
market_demand_per_day = 1000
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

print("[TRIAL 2] Price Floor + Observation (Myopic Best Response)")
print(f"  Duration    : {num_days} days")
print(f"  Price Floor : ${price_floor}")
print()

data = {a: {'prices': [], 'profits': [], 'market_shares': []} for a in airlines}
prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])

for day in range(num_days):
    if day % 250 == 0:
        print(f"  Day {day}/{num_days}  avg price ${prices.mean():.1f}")

    exp_u = np.exp(-alpha * prices)
    shares = exp_u / exp_u.sum()
    demands = shares * market_demand_per_day
    profits = (prices - base_costs) * demands

    for i, a in enumerate(airlines):
        data[a]['prices'].append(prices[i])
        data[a]['profits'].append(profits[i])
        data[a]['market_shares'].append(shares[i] * 100)

    # Strategy: best-response to competitor average, subject to floor
    new_prices = np.zeros(num_airlines)
    for i in range(num_airlines):
        comp_avg = np.mean(np.delete(prices, i))
        # Test: undercut avg, match avg, slight premium
        candidates = np.array([
            comp_avg * 0.97,
            comp_avg,
            comp_avg * 1.03,
            prices[i] * 0.97,
            prices[i],
            prices[i] * 1.03
        ])
        best_p, best_pi = prices[i], -np.inf
        for cp in candidates:
            cp = max(cp, price_floor)
            tp = prices.copy(); tp[i] = cp
            eu = np.exp(-alpha * tp)
            sh = eu / eu.sum()
            pi = (cp - base_costs[i]) * sh[i] * market_demand_per_day
            if pi > best_pi:
                best_pi = pi; best_p = cp
        new_prices[i] = max(best_p, price_floor)
    prices = new_prices

# ── Plotting ────────────────────────────────────────────────
print("\n[TRIAL 2] Creating visualisation …")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=600)
fig.suptitle('Trial 2: Price Floor + Observation (Converges to Floor)', fontsize=14, y=0.98)
days_range = range(num_days)

for i, a in enumerate(airlines):
    ax1.plot(days_range, data[a]['prices'], label=a, color=colors[i], linewidth=1.5)
ax1.axhline(y=price_floor, color='red', ls='--', lw=2, alpha=.6, label=f'Price Floor (${price_floor:.0f})')
ax1.axhline(y=65, color='black', ls=':', lw=1, alpha=.4, label='Cost ($65)')
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
output_dir = "trials/trial2_floor_observation"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"trial2_600dpi.png", dpi=600, bbox_inches='tight', facecolor='white')

final = {"Airline": airlines,
         "Final_Price": [data[a]['prices'][-1] for a in airlines],
         "Avg_Price":   [np.mean(data[a]['prices']) for a in airlines]}
pd.DataFrame(final).to_csv(f"{output_dir}/final_stats.csv", index=False)
print(f"[TRIAL 2] Saved to {output_dir}/\n")
