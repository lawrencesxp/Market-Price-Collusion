"""
PLEASE EXPORT - Rule-Based with Stability Check (10-day window)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(42)

num_days = 50
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
base_costs = np.ones(num_airlines) * 65
alpha = 1.5
market_demand_per_day = 1000
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

print("[PLEASE] Starting rule-based with stability check...")

data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': []} for airline in airlines}
prices = np.array([100, 110, 120, 130, 140])
last_few_prices = [[] for _ in range(num_airlines)]

for day in range(0, num_days):
    if day % 25 == 0:
        print(f"  Day {day}/{num_days}...")
    
    exp_utils = np.exp(-alpha * prices)
    demand_shares = exp_utils / np.sum(exp_utils)
    demands = demand_shares * market_demand_per_day
    
    profits = np.zeros(num_airlines)
    for i, airline in enumerate(airlines):
        profit = (prices[i] - base_costs[i]) * demands[i]
        market_share = demand_shares[i] * 100
        data[airline]['prices'].append(prices[i])
        data[airline]['profits'].append(profit)
        data[airline]['demands'].append(demands[i])
        if len(data[airline]['cumulative_profits']) == 0:
            data[airline]['cumulative_profits'].append(profit)
        else:
            data[airline]['cumulative_profits'].append(data[airline]['cumulative_profits'][-1] + profit)
        profits[i] = profit
        
        last_few_prices[i].append(prices[i])
        if len(last_few_prices[i]) > 10:
            last_few_prices[i].pop(0)
    
    # Strategy: if stable for 10 days, maintain; else adjust slightly
    avg_price = np.mean(prices)
    new_prices = np.zeros(num_airlines)
    
    for i in range(num_airlines):
        if len(last_few_prices[i]) >= 10:
            price_changes = np.abs(np.diff(last_few_prices[i]))
            if np.max(price_changes) < 1.0:
                # Stable - maintain
                new_prices[i] = prices[i] + np.random.normal(0, 0.5)
            else:
                # Unstable - adjust toward average
                new_prices[i] = prices[i] * 0.9 + avg_price * 0.1
        else:
            new_prices[i] = prices[i] + np.random.normal(0, 1)
        
        new_prices[i] = max(new_prices[i], base_costs[i] + 0.01)
        new_prices[i] = min(new_prices[i], 150)
    
    prices = new_prices

print("[PLEASE] Creating visualization...")

fig = plt.figure(figsize=(14, 8), dpi=100)
fig.suptitle('Rule-Based Pricing with 10-Day Stability Check', fontsize=14, y=0.98)

days = list(range(len(data[airlines[0]]['prices'])))

# Plot 1: Price Trajectories
ax1 = plt.subplot(211)

for i, airline in enumerate(airlines):
    ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i], linewidth=2)

ax1.set_title("Price Trajectories", fontsize=12, pad=6)
ax1.set_ylabel("Price ($)", fontsize=11)
ax1.legend(loc='best', fontsize=8, framealpha=0.9, ncol=5, title='Airlines')
ax1.grid(True, alpha=0.15)
ax1.set_xlim(0, num_days)
ax1.set_ylim(60, 180)

# Plot 2: Market Share
ax2 = plt.subplot(212)

for i, airline in enumerate(airlines):
    shares = np.array(data[airline]['demands']) / market_demand_per_day * 100
    ax2.plot(days, shares, label=airline, color=colors[i], linewidth=2)

ax2.set_title("Market Share (%)", fontsize=12, pad=6)
ax2.set_xlabel("Day", fontsize=11)
ax2.set_ylabel("Market Share (%)", fontsize=11)
ax2.grid(True, alpha=0.15)
ax2.set_xlim(0, num_days)
ax2.set_ylim(0, 100)

plt.tight_layout()

output_dir = "exports/RuleBased_StabilitySticking"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/RuleBased_StabilitySticking_300dpi.png", dpi=300, bbox_inches='tight', facecolor='white')

final_data = {"Airline": airlines, "Final_Price": [data[a]['prices'][-1] for a in airlines], 
    "Cumulative_Profit": [data[a]['cumulative_profits'][-1] for a in airlines]}
pd.DataFrame(final_data).to_csv(f"{output_dir}/final_state.csv", index=False)

plt.close()
print("[PLEASE] Export complete!\n")
