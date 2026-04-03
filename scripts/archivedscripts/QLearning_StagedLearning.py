"""
QLEARNING EXPORT - Multi-Agent Q-Learning with Progressive Difficulty
Pre-training (100 days) + Multi-Agent (100 days)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os

np.random.seed(42)

# Configuration
num_days_pre_training = 25
num_days_multi_agent = 25
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
base_costs = np.ones(num_airlines) * 65
alpha = 1.0
market_demand_per_day = 1000
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
baseline_share = 0.05

price_actions = [-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10]
learning_rate = 0.1
discount_factor = 0.9
exploration_rate_initial = 1.0
exploration_decay = 0.99
exploration_min = 0.01

def get_state(airline_idx, prices, profits, rank):
    own_price = prices[airline_idx]
    price_tier = 0 if own_price < 80 else 1 if own_price < 120 else 2
    profit = profits[airline_idx]
    profit_tier = 0 if profit < 0 else 1 if profit < 10000 else 2
    avg_price = np.mean(prices)
    relative_price = 0 if own_price < avg_price * 0.9 else 2 if own_price > avg_price * 1.1 else 1
    return (price_tier, profit_tier, rank, relative_price)

def select_action(q_table, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(len(price_actions))
    return np.argmax(q_table[state]) if any(q_table[state]) else np.random.randint(len(price_actions))

def update_q_table(q_table, state, action, reward, next_state, lr, df):
    current_q = q_table[state][action]
    max_next_q = np.max(q_table[next_state])
    q_table[state][action] = current_q + lr * (reward + df * max_next_q - current_q)

def get_static_price(day, strategy, base_price):
    if strategy == "fixed":
        return base_price
    elif strategy == "gradual_increase":
        return base_price * (1 + 0.001 * day)
    return base_price

print("[QLEARNING] Starting pre-training phase...")

# Initialize Q-tables
q_tables = {airline: defaultdict(lambda: np.zeros(len(price_actions))) for airline in airlines}
exploration_rates = {airline: exploration_rate_initial for airline in airlines}

data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': []} for airline in airlines}
prices = np.array([100, 110, 120, 130, 140])

# Pre-training phase - single agent trains against static competitors
training_airline_idx = 0
training_airline = airlines[training_airline_idx]

for day in range(1, num_days_pre_training + 1):
    if day % 25 == 0:
        print(f"  Pre-training Day {day}/{num_days_pre_training}...")
    
    # Static competitors
    static_prices = np.array([
        get_static_price(day, "fixed", 100),
        get_static_price(day, "gradual_increase", 110),
        get_static_price(day, "fixed", 120),
        get_static_price(day, "fixed", 130)
    ])
    
    prices[0] = np.random.normal(100, 5)
    prices[1:] = static_prices
    
    exp_utils = np.exp(-alpha * prices)
    demand_shares = exp_utils / np.sum(exp_utils)
    demands = demand_shares * market_demand_per_day
    
    profits = (prices - base_costs) * demands
    profit_ranks = np.argsort(np.argsort(-profits))
    
    for i, airline in enumerate(airlines):
        if i < len(prices):
            data[airline]['prices'].append(prices[i])
            data[airline]['profits'].append(profits[i])
            data[airline]['demands'].append(demands[i])
            if len(data[airline]['cumulative_profits']) == 0:
                data[airline]['cumulative_profits'].append(profits[i])
            else:
                data[airline]['cumulative_profits'].append(data[airline]['cumulative_profits'][-1] + profits[i])
    
    # Q-learning for training airline
    state = get_state(training_airline_idx, prices, profits, profit_ranks[training_airline_idx])
    action = select_action(q_tables[training_airline], state, exploration_rates[training_airline])
    price_change = price_actions[action]
    reward = profits[training_airline_idx]
    
    exploration_rates[training_airline] *= exploration_decay
    exploration_rates[training_airline] = max(exploration_rates[training_airline], exploration_min)

print("[QLEARNING] Starting multi-agent phase...")

# Multi-agent phase - all agents compete
for day in range(1, num_days_multi_agent + 1):
    if day % 25 == 0:
        print(f"  Multi-agent Day {day}/{num_days_multi_agent}...")
    
    exp_utils = np.exp(-alpha * prices)
    demand_shares = exp_utils / np.sum(exp_utils)
    demands = demand_shares * market_demand_per_day
    profits = (prices - base_costs) * demands
    profit_ranks = np.argsort(np.argsort(-profits))
    
    for i, airline in enumerate(airlines):
        data[airline]['prices'].append(prices[i])
        data[airline]['profits'].append(profits[i])
        data[airline]['demands'].append(demands[i])
        if len(data[airline]['cumulative_profits']) == 0:
            data[airline]['cumulative_profits'].append(profits[i])
        else:
            data[airline]['cumulative_profits'].append(data[airline]['cumulative_profits'][-1] + profits[i])
    
    new_prices = np.zeros(num_airlines)
    for i, airline in enumerate(airlines):
        state = get_state(i, prices, profits, profit_ranks[i])
        action = select_action(q_tables[airline], state, exploration_rates[airline])
        new_price = prices[i] * (1 + price_actions[action])
        new_price = max(new_price, base_costs[i] + 0.01)
        new_price = min(new_price, 180)
        new_prices[i] = new_price
        exploration_rates[airline] *= exploration_decay
        exploration_rates[airline] = max(exploration_rates[airline], exploration_min)
    
    prices = new_prices

print("[QLEARNING] Creating visualization...")

fig = plt.figure(figsize=(14, 8), dpi=100)
fig.suptitle('Q-Learning with Staged Learning', fontsize=14, y=0.98)

days = list(range(len(data[airlines[0]]['prices'])))

# Plot 1: Price Trajectories
ax1 = plt.subplot(211)
if num_days_pre_training > 0:
    ax1.axvline(x=num_days_pre_training, color='black', linestyle=':', linewidth=1, alpha=0.5)

for i, airline in enumerate(airlines):
    ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i], linewidth=2)

ax1.set_title("Price Trajectories", fontsize=12, pad=6)
ax1.set_ylabel("Price ($)", fontsize=11)
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=5, title='Airlines')
ax1.grid(True, alpha=0.15)
ax1.set_xlim(0, num_days_pre_training + num_days_multi_agent)
ax1.set_ylim(60, 180)

# Plot 2: Market Share
ax2 = plt.subplot(212)

for i, airline in enumerate(airlines):
    market_shares = np.array(data[airline]['demands']) / market_demand_per_day * 100
    ax2.plot(days, market_shares, label=airline, color=colors[i], linewidth=2)

ax2.set_title("Market Share (%)", fontsize=12, pad=6)
ax2.set_xlabel("Day", fontsize=11)
ax2.set_ylabel("Market Share (%)", fontsize=11)
ax2.grid(True, alpha=0.15)
ax2.set_xlim(0, num_days_pre_training + num_days_multi_agent)
ax2.set_ylim(0, 100)

plt.tight_layout()

output_dir = "exports/QLearning_StagedLearning"
os.makedirs(output_dir, exist_ok=True)

print(f"[QLEARNING] Saving to {output_dir}/...")
plt.savefig(f"{output_dir}/QLearning_StagedLearning_300dpi.png", dpi=300, bbox_inches='tight', facecolor='white')
print(f"  [OK] QLearning_StagedLearning_300dpi.png (300 DPI)")

final_data = {
    "Airline": airlines,
    "Final_Price": [data[a]['prices'][-1] for a in airlines],
    "Final_Profit": [data[a]['profits'][-1] for a in airlines],
    "Cumulative_Profit": [data[a]['cumulative_profits'][-1] for a in airlines]
}
pd.DataFrame(final_data).to_csv(f"{output_dir}/final_state.csv", index=False)
print(f"  [OK] final_state.csv")

plt.close()

print(f"\n[QLEARNING] Export complete!\n")
