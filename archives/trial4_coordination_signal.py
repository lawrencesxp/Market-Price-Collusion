"""
TRIAL 4: COORDINATION SIGNAL (MOVING AVERAGE + STABILITY PREFERENCE)
Each airline tracks a 10-day moving average of market prices as a COORDINATION SIGNAL.
Airlines prefer to stay near this signal (stability-seeking behavior) while maintaining
some profit-seeking. This simulates implicit coordination without explicit collusion.
Purpose: Test if shared observation of market trend can lead to self-coordination.
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

ma_window = 10  # 10-day moving average
stability_weight = 0.6  # How much to value staying near market average (60% stability, 40% profit)

print("[TRIAL 4] Coordination Signal (Moving Average + Stability Preference)")
print(f"  Duration: {num_days} days")
print(f"  Airlines: {num_airlines}")
print(f"  Strategy: Track 10-day moving average of market prices as coordination signal")
print(f"  Stability Weight: {stability_weight*100:.0f}% (prefer stability, {(1-stability_weight)*100:.0f}% profit)")
print()

data = {airline: {'prices': [], 'profits': [], 'demands': [], 'market_shares': []} for airline in airlines}
prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])

# Track price history for moving average
price_history = [prices.copy()]

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
    
    # Calculate moving average of historical market prices (all airlines' prices)
    if len(price_history) >= ma_window:
        recent_prices = np.array(price_history[-ma_window:])  # Last 10 periods
        market_average = np.mean(recent_prices)  # Average of all recent prices
    else:
        market_average = np.mean(prices)
    
    # Strategy: Balance between profit maximization and staying near market average
    new_prices = np.zeros(num_airlines)
    
    for i in range(num_airlines):
        # Profit-seeking candidates (like before)
        profit_candidates = np.array([
            prices[i] * 0.97,
            prices[i],
            prices[i] * 1.03
        ])
        
        # Stability-seeking candidate (stay near market average)
        stability_candidate = market_average
        
        # All candidates to evaluate
        candidates = np.append(profit_candidates, stability_candidate)
        
        best_price = prices[i]
        best_score = -np.inf
        
        for candidate_price in candidates:
            test_prices = prices.copy()
            test_prices[i] = candidate_price
            
            # Calculate profit
            exp_utils_test = np.exp(-alpha * test_prices)
            shares_test = exp_utils_test / np.sum(exp_utils_test)
            demands_test = shares_test * market_demand_per_day
            profit_test = (test_prices[i] - base_costs[i]) * demands_test[i]
            
            # Normalize profit (0-100 scale approx)
            normalized_profit = max(0, min(profit_test / 200, 1.0))  # Rough normalization
            
            # Stability score: how close to market average
            price_diff = abs(candidate_price - market_average)
            normalized_stability = max(0, 1 - (price_diff / 100))  # Higher is better
            
            # Composite score
            score = (stability_weight * normalized_stability) + ((1-stability_weight) * normalized_profit)
            
            if score > best_score:
                best_score = score
                best_price = candidate_price
        
        new_prices[i] = best_price
        new_prices[i] = max(new_prices[i], base_costs[i] + 0.01)  # Cost floor
    
    price_history.append(prices.copy())
    prices = new_prices

print("\n[TRIAL 4] Creating visualization...")

fig = plt.figure(figsize=(14, 8), dpi=100)
fig.suptitle('Trial 4: Coordination Signal (Moving Average + Stability Seeking)', fontsize=14, y=0.98)

days = list(range(len(data[airlines[0]]['prices'])))

# Plot 1: Price Trajectories
ax1 = plt.subplot(211)
for i, airline in enumerate(airlines):
    ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i], linewidth=2)

# Also show market average for reference
market_prices_all = []
for day in days:
    day_prices = [data[a]['prices'][day] for a in airlines]
    market_prices_all.append(np.mean(day_prices))

ax1.plot(days, market_prices_all, 'k--', linewidth=1.5, alpha=0.5, label='Market Avg')
ax1.set_title("Price Trajectories (Stabilizing Around Market Average)", fontsize=12, pad=6)
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

output_dir = "trials/trial4_coordination_signal"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/trial4_300dpi.png", dpi=300, bbox_inches='tight', facecolor='white')

final_data = {
    "Airline": airlines,
    "Final_Price": [data[a]['prices'][-1] for a in airlines],
    "Final_Profit": [data[a]['profits'][-1] for a in airlines],
    "Avg_Price": [np.mean(data[a]['prices']) for a in airlines],
    "Price_Variance": [np.var(data[a]['prices']) for a in airlines],
    "Avg_Market_Share": [np.mean(data[a]['market_shares']) for a in airlines],
}
pd.DataFrame(final_data).to_csv(f"{output_dir}/final_stats.csv", index=False)

history_data = {"Day": list(range(num_days)), "Market_Avg": market_prices_all}
for airline in airlines:
    history_data[f"{airline}_Price"] = data[airline]['prices']

pd.DataFrame(history_data).to_csv(f"{output_dir}/full_history.csv", index=False)

print(f"[TRIAL 4] Saved to {output_dir}/")
print(f"[TRIAL 4] Export complete!\n")
