"""
QLEARNING2 EXPORT - Q-Learning with Pre-training + Multi-Agent (100+100 days)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os

np.random.seed(42)

num_days_pre = 25
num_days_multi = 25
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
base_costs = np.ones(num_airlines) * 65
alpha = 1.5
mkt_demand = 1000
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

price_actions = [-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10]
lr, df, eps_init, eps_dec, eps_min = 0.1, 0.9, 1.0, 0.99, 0.01

print("[QLEARNING2] Starting pre-training phase...")

q_tables = {a: defaultdict(lambda: np.zeros(len(price_actions))) for a in airlines}
eps_rates = {a: eps_init for a in airlines}

data = {a: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': []} for a in airlines}
prices = np.array([100, 110, 120, 130, 140])

def get_state(idx, ps, pfs, rank):
    own = ps[idx]
    pt = 0 if own < 80 else 1 if own < 120 else 2
    pft = 0 if pfs[idx] < 0 else 1 if pfs[idx] < 10000 else 2
    rp = 0 if own < np.mean(ps)*0.9 else 2 if own > np.mean(ps)*1.1 else 1
    return (pt, pft, rank, rp)

# Pre-training
for day in range(num_days_pre):
    if day % 25 == 0:
        print(f"  Pre-training Day {day}/{num_days_pre}...")
    
    exp_utils = np.exp(-alpha * prices)
    dem_shares = exp_utils / np.sum(exp_utils)
    demands = dem_shares * mkt_demand
    profits = (prices - base_costs) * demands
    ranks = np.argsort(np.argsort(-profits))
    
    for i, a in enumerate(airlines):
        data[a]['prices'].append(prices[i])
        data[a]['profits'].append(profits[i])
        data[a]['demands'].append(demands[i])
        if len(data[a]['cumulative_profits']) == 0:
            data[a]['cumulative_profits'].append(profits[i])
        else:
            data[a]['cumulative_profits'].append(data[a]['cumulative_profits'][-1] + profits[i])
    
    prices = prices + np.random.normal(0, 1, num_airlines)
    prices = np.maximum(prices, base_costs + 0.01)
    prices = np.minimum(prices, 150)
    
    for a in airlines:
        eps_rates[a] *= eps_dec
        eps_rates[a] = max(eps_rates[a], eps_min)

print("[QLEARNING2] Starting multi-agent phase...")

# Multi-agent
for day in range(num_days_multi):
    if day % 25 == 0:
        print(f"  Multi-agent Day {day}/{num_days_multi}...")
    
    exp_utils = np.exp(-alpha * prices)
    dem_shares = exp_utils / np.sum(exp_utils)
    demands = dem_shares * mkt_demand
    profits = (prices - base_costs) * demands
    ranks = np.argsort(np.argsort(-profits))
    
    for i, a in enumerate(airlines):
        data[a]['prices'].append(prices[i])
        data[a]['profits'].append(profits[i])
        data[a]['demands'].append(demands[i])
        if len(data[a]['cumulative_profits']) == 0:
            data[a]['cumulative_profits'].append(profits[i])
        else:
            data[a]['cumulative_profits'].append(data[a]['cumulative_profits'][-1] + profits[i])
    
    new_prices = np.zeros(num_airlines)
    for i, a in enumerate(airlines):
        state = get_state(i, prices, profits, ranks[i])
        if np.random.random() < eps_rates[a]:
            action = np.random.randint(len(price_actions))
        else:
            action = np.argmax(q_tables[a][state])
        
        new_price = prices[i] * (1 + price_actions[action])
        new_price = max(new_price, base_costs[i] + 0.01)
        new_price = min(new_price, 150)
        new_prices[i] = new_price
        eps_rates[a] *= eps_dec
        eps_rates[a] = max(eps_rates[a], eps_min)
    
    prices = new_prices

print("[QLEARNING2] Creating visualization...")

fig = plt.figure(figsize=(14, 8), dpi=100)
fig.suptitle('Q-Learning with Random Walk Pre-training', fontsize=14, y=0.98)

days = list(range(len(data[airlines[0]]['prices'])))

# Plot 1: Price Trajectories
ax1 = plt.subplot(211)
if num_days_pre > 0:
    ax1.axvline(x=num_days_pre, color='black', linestyle=':', linewidth=1, alpha=0.5)

for i, airline in enumerate(airlines):
    ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i], linewidth=2)

ax1.set_title("Price Trajectories", fontsize=12, pad=6)
ax1.set_ylabel("Price ($)", fontsize=11)
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=5, title='Airlines')
ax1.grid(True, alpha=0.15)
ax1.set_xlim(0, num_days_pre + num_days_multi)
ax1.set_ylim(60, 180)

# Plot 2: Market Share
ax2 = plt.subplot(212)

for i, airline in enumerate(airlines):
    shares = np.array(data[airline]['demands']) / mkt_demand * 100
    ax2.plot(days, shares, label=airline, color=colors[i], linewidth=2)

ax2.set_title("Market Share (%)", fontsize=12, pad=6)
ax2.set_xlabel("Day", fontsize=11)
ax2.set_ylabel("Market Share (%)", fontsize=11)
ax2.grid(True, alpha=0.15)
ax2.set_xlim(0, num_days_pre + num_days_multi)
ax2.set_ylim(0, 100)

plt.tight_layout()

output_dir = "exports/QLearning_RandomWalk"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/QLearning_RandomWalk_300dpi.png", dpi=300, bbox_inches='tight', facecolor='white')
print(f"[QLEARNING2] Saved QLearning_RandomWalk_300dpi.png")

final_data = {"Airline": airlines, "Final_Price": [data[a]['prices'][-1] for a in airlines], 
    "Cumulative_Profit": [data[a]['cumulative_profits'][-1] for a in airlines]}
pd.DataFrame(final_data).to_csv(f"{output_dir}/final_state.csv", index=False)

plt.close()
print("[QLEARNING2] Export complete!\n")
