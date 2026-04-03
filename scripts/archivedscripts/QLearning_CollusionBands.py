"""
LOVEOFGOD EXPORT - Q-Learning with Enhanced Collusion Rewards
Runs 300-day simulation and exports high-resolution visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import defaultdict
import os

np.random.seed(17)

# Configuration
num_days = 50
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
base_costs = np.ones(num_airlines) * 75 
alpha = 1.5
market_demand_per_day = 1000
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
min_market_share = 0.07

price_actions = [-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10]
learning_rate = 0.1
discount_factor = 0.9
exploration_rate_initial = 1.0
exploration_decay = 0.99
exploration_min = 0.05

collusion_band_width = 0.075
collusion_reward_bonus = 8000
price_premium_threshold = 100

def get_state(airline_idx, prices, profits, ranks):
    own_price = prices[airline_idx]
    price_tier = 0 if own_price < 85 else 1 if own_price < 110 else 2
    profit = profits[airline_idx]
    profit_tier = 0 if profit < 0 else 1 if profit < 10000 else 2
    rank = ranks[airline_idx]
    avg_price = np.mean(prices)
    relative_price = 0 if own_price < avg_price * 0.9 else 2 if own_price > avg_price * 1.1 else 1
    return (price_tier, profit_tier, rank, relative_price)

def is_in_collusion_band(price, avg_price):
    lower = avg_price * (1 - collusion_band_width)
    upper = avg_price * (1 + collusion_band_width)
    return lower <= price <= upper

def calculate_collusion_bonus(price, all_prices, profits):
    avg_price = np.mean(all_prices)
    if is_in_collusion_band(price, avg_price):
        others_in_band = sum(1 for p in all_prices if is_in_collusion_band(p, avg_price)) - 1
        band_factor = others_in_band / (len(all_prices) - 1) if len(all_prices) > 1 else 0
        if band_factor > 0.8:
            band_factor *= 1.5
        profit_factor = min(1.0, max(0.0, np.mean(profits) / 10000))
        price_premium = 1.0
        if avg_price > price_premium_threshold:
            premium_ratio = (avg_price - price_premium_threshold) / 50
            price_premium = 1.0 + min(2.0, premium_ratio)
        return collusion_reward_bonus * band_factor * profit_factor * price_premium
    return 0

def adjust_demand_shares(demand_shares, min_share=min_market_share):
    total_min_share = min_share * len(demand_shares)
    if total_min_share > 0.95:
        min_share = 0.95 / len(demand_shares)
        total_min_share = 0.95
    remaining_share = 1.0 - total_min_share
    if sum(demand_shares) > 0:
        normalized_shares = demand_shares / sum(demand_shares)
    else:
        normalized_shares = np.ones_like(demand_shares) / len(demand_shares)
    adjusted_shares = min_share + normalized_shares * remaining_share
    adjusted_shares = adjusted_shares / sum(adjusted_shares)
    return adjusted_shares

class QLearningAirline:
    def __init__(self, airline_id):
        self.airline_id = airline_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate_initial
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.q_table = defaultdict(lambda: np.zeros(len(price_actions)))
        self.last_state = None
        self.last_action = None
    
    def select_action(self, state):
        if np.random.random() < self.exploration_rate:
            return np.random.randint(len(price_actions))
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
    
    def get_action_for_price(self, current_price, action_idx):
        price_change = price_actions[action_idx]
        new_price = current_price * (1 + price_change)
        new_price = max(new_price, base_costs[0] + 0.01)
        new_price = min(new_price, 180)
        return new_price

# Initialize
print("[LOVEOFGOD] Starting Q-Learning simulation (300 days)...")
q_learning_agents = {airline: QLearningAirline(i) for i, airline in enumerate(airlines)}
data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': [], 
                  'market_shares': []} for airline in airlines}
prices = np.array([100, 110, 120, 130, 140])

# Initial conditions
exp_utils = np.exp(-alpha * prices)
raw_demand_shares = exp_utils / np.sum(exp_utils)
demand_shares = adjust_demand_shares(raw_demand_shares)
demands = demand_shares * market_demand_per_day
for i, airline in enumerate(airlines):
    price = prices[i]
    demand = demands[i]
    cost = base_costs[i]
    profit = (price - cost) * demand
    market_share = demand_shares[i] * 100
    data[airline]['prices'].append(price)
    data[airline]['profits'].append(profit)
    data[airline]['demands'].append(demand)
    data[airline]['cumulative_profits'].append(profit)
    data[airline]['market_shares'].append(market_share)

# Simulation loop
for day in range(1, num_days + 1):
    if day % 50 == 0:
        print(f"  Day {day}/{num_days}...")
    
    exp_utils = np.exp(-alpha * prices)
    raw_demand_shares = exp_utils / np.sum(exp_utils)
    demand_shares = adjust_demand_shares(raw_demand_shares)
    demands = demand_shares * market_demand_per_day
    profits = np.zeros(num_airlines)
    
    for i, airline in enumerate(airlines):
        price = prices[i]
        demand = demands[i]
        cost = base_costs[i]
        profit = (price - cost) * demand
        market_share = demand_shares[i] * 100
        data[airline]['prices'].append(price)
        data[airline]['profits'].append(profit)
        data[airline]['demands'].append(demand)
        data[airline]['market_shares'].append(market_share)
        if len(data[airline]['cumulative_profits']) == 0:
            data[airline]['cumulative_profits'].append(profit)
        else:
            data[airline]['cumulative_profits'].append(data[airline]['cumulative_profits'][-1] + profit)
        profits[i] = profit
    
    profit_ranks = np.argsort(np.argsort(-profits))
    new_prices = np.zeros(num_airlines)
    best_airline_idx = np.argmax(profits)
    best_price = prices[best_airline_idx]
    
    for i, airline in enumerate(airlines):
        agent = q_learning_agents[airline]
        state = get_state(i, prices, profits, profit_ranks)
        
        if agent.last_state is not None and agent.last_action is not None:
            reward = profits[i]
            reward += calculate_collusion_bonus(prices[i], prices, profits)
            agent.learn(agent.last_state, agent.last_action, reward, state)
        
        action_idx = agent.select_action(state)
        new_price = agent.get_action_for_price(prices[i], action_idx)
        if np.random.random() < agent.exploration_rate:
            new_price += np.random.normal(0, 2.0)
            new_price = max(new_price, base_costs[i] + 0.01)
            new_price = min(new_price, 180)
        
        new_prices[i] = new_price
        agent.last_state = state
        agent.last_action = action_idx
    
    prices = new_prices

print("[LOVEOFGOD] Creating visualization...")

print("[LOVEOFGOD] Creating visualization...")

fig = plt.figure(figsize=(14, 8), dpi=100)
fig.suptitle('Q-Learning with Collusion Bands', fontsize=14, y=0.98)

days = list(range(len(data[airlines[0]]['prices'])))

# Plot 1: Price Trajectories
ax1 = plt.subplot(211)

for i, airline in enumerate(airlines):
    ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i], linewidth=2)

ax1.set_title("Price Trajectories", fontsize=12, pad=6)
ax1.set_ylabel("Price ($)", fontsize=11)
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=5, title='Airlines')
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

# Save exports
output_dir = "exports/QLearning_CollusionBands"
os.makedirs(output_dir, exist_ok=True)

print(f"[LOVEOFGOD] Saving to {output_dir}/...")
plt.savefig(f"{output_dir}/QLearning_CollusionBands_300dpi.png", dpi=300, bbox_inches='tight', facecolor='white')
print(f"  [OK] QLearning_CollusionBands_300dpi.png (300 DPI)")

# Save final state CSV
final_data = {
    "Airline": airlines,
    "Final_Price": [data[a]['prices'][-1] for a in airlines],
    "Final_Profit": [data[a]['profits'][-1] for a in airlines],
    "Market_Share": [data[a]['market_shares'][-1] for a in airlines],
    "Cumulative_Profit": [data[a]['cumulative_profits'][-1] for a in airlines]
}
pd.DataFrame(final_data).to_csv(f"{output_dir}/final_state.csv", index=False)
print(f"  [OK] final_state.csv")

# Save full data history
history_data = {
    "Day": days,
}
for airline in airlines:
    history_data[f"{airline}_Price"] = data[airline]['prices']
    history_data[f"{airline}_Profit"] = data[airline]['profits']
    history_data[f"{airline}_Market_Share"] = data[airline]['market_shares']

pd.DataFrame(history_data).to_csv(f"{output_dir}/full_history.csv", index=False)
print(f"  [OK] full_history.csv")

plt.close()

print(f"\n[LOVEOFGOD] Export complete!\n")
