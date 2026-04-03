"""
SUSTAINPROFIT EXPORT - User-Controlled Airline_5 vs AI Rule-Based Competitors
Simulating user following a "sustain profit" strategy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(17)

num_days = 50
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
base_costs = np.ones(num_airlines) * 65
alpha = 1.5
mkt_demand = 1000
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

print("[SUSTAINPROFIT] Starting user-controlled simulation (100 days)...")

data = {a: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': [], 'market_shares': []} for a in airlines}
prices = np.array([100, 110, 120, 130, 140])

user_airline_idx = 4  # Airline_5

for day in range(num_days):
    if day % 25 == 0:
        print(f"  Day {day}/{num_days}...")
    
    exp_utils = np.exp(-alpha * prices)
    dem_shares = exp_utils / np.sum(exp_utils)
    demands = dem_shares * mkt_demand
    profits = (prices - base_costs) * demands
    ranks = np.argsort(np.argsort(-profits))
    
    for i, a in enumerate(airlines):
        mkt_share = dem_shares[i] * 100
        data[a]['prices'].append(prices[i])
        data[a]['profits'].append(profits[i])
        data[a]['demands'].append(demands[i])
        data[a]['market_shares'].append(mkt_share)
        if len(data[a]['cumulative_profits']) == 0:
            data[a]['cumulative_profits'].append(profits[i])
        else:
            data[a]['cumulative_profits'].append(data[a]['cumulative_profits'][-1] + profits[i])
    
    # User strategy for Airline_5: sustain current profit level
    target_profit = np.mean(profits[:4]) * 0.95
    current_profit = profits[user_airline_idx]
    
    if current_profit < target_profit * 0.9:
        # Reduce price to grab market share
        prices[user_airline_idx] = prices[user_airline_idx] * 0.97
    elif current_profit > target_profit * 1.1:
        # Increase price to maximize margins
        prices[user_airline_idx] = prices[user_airline_idx] * 1.02
    else:
        # Maintain price
        prices[user_airline_idx] = prices[user_airline_idx] + np.random.normal(0, 0.5)
    
    prices[user_airline_idx] = max(prices[user_airline_idx], base_costs[user_airline_idx] + 0.01)
    prices[user_airline_idx] = min(prices[user_airline_idx], 180)
    
    # AI competitors: simple rules
    avg_price = np.mean(prices[:4])
    for i in range(4):
        if ranks[i] >= 2:  # Bottom performers
            prices[i] = avg_price * 0.95
        else:
            prices[i] = avg_price * 1.02
        
        prices[i] = max(prices[i], base_costs[i] + 0.01)
        prices[i] = min(prices[i], 180)

print("[SUSTAINPROFIT] Creating visualization...")

fig = plt.figure(figsize=(14, 8), dpi=100)
fig.suptitle('User-Controlled Airline Strategy (Sustain Profit) vs. AI Competitors', fontsize=14, y=0.98)

days = list(range(len(data[airlines[0]]['prices'])))

# Plot 1: Price Trajectories
ax1 = plt.subplot(211)

for i, airline in enumerate(airlines):
    style = 'solid' if i < 4 else 'dashed'
    width = 2 if i < 4 else 2.5
    ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i], 
             linewidth=width, linestyle=style)

ax1.set_title("Price Trajectories (User = Airline_5, Dashed)", fontsize=12, pad=6)
ax1.set_ylabel("Price ($)", fontsize=11)
ax1.legend(loc='best', fontsize=8, framealpha=0.9, ncol=5, title='Airlines')
ax1.grid(True, alpha=0.15)
ax1.set_xlim(0, num_days)
ax1.set_ylim(60, 180)

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

output_dir = "exports/Interactive_ProfitTargeting"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/Interactive_ProfitTargeting_300dpi.png", dpi=300, bbox_inches='tight', facecolor='white')

final_data = {"Airline": airlines, "Final_Price": [data[a]['prices'][-1] for a in airlines], 
    "Cumulative_Profit": [data[a]['cumulative_profits'][-1] for a in airlines],
    "Market_Share": [data[a]['market_shares'][-1] for a in airlines]}
pd.DataFrame(final_data).to_csv(f"{output_dir}/final_state.csv", index=False)

plt.close()
print("[SUSTAINPROFIT] Export complete!\n")
