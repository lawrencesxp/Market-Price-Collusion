"""
TRIAL 3: SIMPLE MEMORY (BEST RESPONSE)
No external rules. Each airline remembers competitors' prices from last period
and plays best response to that observed signal. This is reactive learning.
Purpose: Test if observation alone + myopic best response leads to coordination.
Expected: May converge to something via oscillation, but without enforcement mechanism.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(42)

num_days = 100
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
base_costs = np.ones(num_airlines) * 65
alpha = 1.5
market_demand_per_day = 1000
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

print("[TRIAL 3] Simple Memory - Myopic Best Response")
print(f"  Duration: {num_days} days")
print(f"  Airlines: {num_airlines}")
print(f"  Strategy: Best response to observed competitor prices (last period)")
print(f"  No external constraint, purely reactive learning")
print()

data = {airline: {'prices': [], 'profits': [], 'demands': [], 'market_shares': []} for airline in airlines}
prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
prev_prices = prices.copy()

for day in range(num_days):
    if day % 25 == 0:
        print(f"  Day {day}/{num_days}...")
    
    # Logit demand model
    exp_utils = np.exp(-alpha * prices)
    demand_shares = exp_utils / np.sum(exp_utils)
    demands = demand_shares * market_demand_per_day
    profits = (prices - base_costs) * demands
    
    # Record data
    for i, airline in enumerate(airlines):
        data[airline]['prices'].append(prices[i])
        data[airline]['profits'].append(profits[i])
        data[airline]['demands'].append(demands[i])
        data[airline]['market_shares'].append(demand_shares[i] * 100)
    
    # Strategy: Best response to competitors' LAST PERIOD prices
    # Observe what competitors did and optimize against that
    new_prices = np.zeros(num_airlines)
    
    for i in range(num_airlines):
        # Get other airlines' prices from last period
        competitor_prices = np.delete(prev_prices, i)
        avg_competitor_price = np.mean(competitor_prices)
        
        # Test different prices against observed competitor behavior
        candidates = np.array([
            avg_competitor_price * 0.98,  # undercut average
            avg_competitor_price,          # match average
            avg_competitor_price * 1.02   # slight premium
        ])
        
        best_price = prices[i]
        best_profit = profits[i]
        
        for candidate_price in candidates:
            test_prices = prices.copy()
            test_prices[i] = candidate_price
            
            exp_utils_test = np.exp(-alpha * test_prices)
            shares_test = exp_utils_test / np.sum(exp_utils_test)
            demands_test = shares_test * market_demand_per_day
            profit_test = (test_prices[i] - base_costs[i]) * demands_test[i]
            
            if profit_test > best_profit:
                best_profit = profit_test
                best_price = candidate_price
        
        new_prices[i] = best_price
        # Cost floor only
        new_prices[i] = max(new_prices[i], base_costs[i] + 0.01)
    
    prev_prices = prices.copy()
    prices = new_prices

print("\n[TRIAL 3] Creating visualization...")

fig = plt.figure(figsize=(14, 8), dpi=100)
fig.suptitle('Trial 3: Simple Memory (Myopic Best Response to Observed Prices)', fontsize=14, y=0.98)

days = list(range(len(data[airlines[0]]['prices'])))

# Plot 1: Price Trajectories
ax1 = plt.subplot(211)
for i, airline in enumerate(airlines):
    ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i], linewidth=2)

ax1.axhline(y=65, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Cost Floor')
ax1.set_title("Price Trajectories (Reactive Learning - May Oscillate)", fontsize=12, pad=6)
ax1.set_ylabel("Price ($)", fontsize=11)
ax1.legend(loc='best', fontsize=8, framealpha=0.9, ncol=5, title='Airlines')
ax1.grid(True, alpha=0.15)
ax1.set_xlim(0, num_days)

# Plot 2: Market Share
ax2 = plt.subplot(212)
for i, airline in enumerate(airlines):
    ax2.plot(days, data[airline]['market_shares'], label=airline, color=colors[i], linewidth=2)

ax2.set_title("Market Share (%)", fontsize=12, pad=6)
ax2.set_xlabel("Day", fontsize=11)
ax2.set_ylabel("Market Share (%)", fontsize=11)
ax2.grid(True, alpha=0.15)
ax2.set_xlim(0, num_days)
ax2.set_ylim(0, 100)

plt.tight_layout()

output_dir = "trials/trial3_simple_memory"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/trial3_300dpi.png", dpi=300, bbox_inches='tight', facecolor='white')

final_data = {
    "Airline": airlines,
    "Final_Price": [data[a]['prices'][-1] for a in airlines],
    "Final_Profit": [data[a]['profits'][-1] for a in airlines],
    "Avg_Price": [np.mean(data[a]['prices']) for a in airlines],
    "Price_Variance": [np.var(data[a]['prices']) for a in airlines],
    "Avg_Market_Share": [np.mean(data[a]['market_shares']) for a in airlines],
}
pd.DataFrame(final_data).to_csv(f"{output_dir}/final_stats.csv", index=False)

history_data = {"Day": list(range(num_days))}
for airline in airlines:
    history_data[f"{airline}_Price"] = data[airline]['prices']

pd.DataFrame(history_data).to_csv(f"{output_dir}/full_history.csv", index=False)

print(f"[TRIAL 3] Saved to {output_dir}/")
print(f"[TRIAL 3] Export complete!\n")
