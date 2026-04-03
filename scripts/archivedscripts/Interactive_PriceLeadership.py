"""
USER EXPORT - Alternative User Control Strategy
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
mkt_demand = 1000
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

print("[USER] Starting alternative user control simulation (100 days)...")

data = {a: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': [], 'market_shares': []} for a in airlines}
prices = np.array([100, 110, 120, 130, 140])

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
    
    # User strategy: be the lowest price competitor (capture market share)
    min_comp_price = np.min(prices[:4])
    
    prices[4] = min_comp_price * 0.98  # Undercut competitors
    prices[4] = max(prices[4], base_costs[4] + 0.01)
    prices[4] = min(prices[4], 180)
    
    # AI competitors adapt
    avg_price = np.mean(prices[:4])
    for i in range(4):
        if prices[4] < avg_price * 0.8:
            prices[i] = prices[i] * 1.01  # Raise price if undercut
        else:
            prices[i] = avg_price + np.random.normal(0, 1)
        
        prices[i] = max(prices[i], base_costs[i] + 0.01)
        prices[i] = min(prices[i], 180)

print("[USER] Creating visualization...")

fig = plt.figure(figsize=(14, 8), dpi=100)
fig.suptitle('User-Controlled Price Leadership Strategy (Lowest Price Competitor)', fontsize=14, y=0.98)

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
    ax2.plot(days, data[airline]['market_shares'], label=airline, color=colors[i], linewidth=2)

ax2.set_title("Market Share (%)", fontsize=12, pad=6)
ax2.set_xlabel("Day", fontsize=11)
ax2.set_ylabel("Market Share (%)", fontsize=11)
ax2.grid(True, alpha=0.15)
ax2.set_xlim(0, num_days)
ax2.set_ylim(0, 100)

plt.tight_layout()

output_dir = "exports/Interactive_PriceLeadership"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/Interactive_PriceLeadership_300dpi.png", dpi=300, bbox_inches='tight', facecolor='white')

final_data = {"Airline": airlines, "Final_Price": [data[a]['prices'][-1] for a in airlines], 
    "Cumulative_Profit": [data[a]['cumulative_profits'][-1] for a in airlines],
    "Market_Share": [data[a]['market_shares'][-1] for a in airlines]}
pd.DataFrame(final_data).to_csv(f"{output_dir}/final_state.csv", index=False)

plt.close()
print("[USER] Export complete!\n")
