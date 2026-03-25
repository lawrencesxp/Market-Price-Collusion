"""
TRIAL 5: Q-LEARNING WITH PROFIT + COLLUSION INCENTIVE
Simple Q-Learning where airlines learn to maximize a composite reward:
  - Immediate profit from the sale
  - Bonus for staying within X% of competitors' average price (collusion incentive)
This tests whether learning alone (without explicit punishment) can discover collusion.
Purpose: Demonstrate emergence of coordination through reinforcement learning.
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

# Q-Learning parameters
price_bins = np.linspace(65, 180, 20)  # 20 price levels to choose from
learning_rate = 0.3
discount_factor = 0.9
epsilon = 0.1  # Exploration rate

# Reward structure
collusion_bonus_weight = 0.4  # 40% of reward from "staying coordinated"
collusion_tolerance = 0.1  # ±10% of competitors' average acceptable

print("[TRIAL 5] Q-Learning with Collusion Incentive")
print(f"  Duration: {num_days} days")
print(f"  Airlines: {num_airlines}")
print(f"  Price levels: {len(price_bins)}")
print(f"  Learning rate: {learning_rate}")
print(f"  Collusion bonus weight: {collusion_bonus_weight*100:.0f}%")
print(f"  Collusion tolerance: ±{collusion_tolerance*100:.0f}% of market average")
print()

data = {airline: {'prices': [], 'profits': [], 'demands': [], 'market_shares': []} for airline in airlines}

# Initialize Q-tables for each airline: Q[state_bin][action_bin]
# State: competitor average price bin
# Action: price level to set
q_tables = [np.zeros((len(price_bins), len(price_bins))) for _ in range(num_airlines)]

# Initial prices
prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])

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
    
    # Q-Learning update for each airline
    new_prices = np.zeros(num_airlines)
    
    for i in range(num_airlines):
        # Get state: average price of competitors
        competitor_prices = np.delete(prices, i)
        competitor_avg = np.mean(competitor_prices)
        state_bin = np.argmin(np.abs(price_bins - competitor_avg))
        
        # Choose action (price) using epsilon-greedy
        if np.random.random() < epsilon:
            # Explore: random action
            action_bin = np.random.randint(len(price_bins))
        else:
            # Exploit: best Q-value
            action_bin = np.argmax(q_tables[i][state_bin, :])
        
        candidate_price = price_bins[action_bin]
        
        # Calculate reward from this action
        test_prices = prices.copy()
        test_prices[i] = candidate_price
        
        exp_utils_test = np.exp(-alpha * test_prices)
        shares_test = exp_utils_test / np.sum(exp_utils_test)
        demands_test = shares_test * market_demand_per_day
        profit_test = (test_prices[i] - base_costs[i]) * demands_test[i]
        
        # Normalize profit to 0-100 scale
        profit_reward = max(0, min(profit_test / 500, 1.0)) * 100
        
        # Collusion bonus: reward for staying within tolerance of competitor average
        price_diff_pct = abs(candidate_price - competitor_avg) / competitor_avg
        
        if price_diff_pct <= collusion_tolerance:
            collusion_reward = 100  # Max bonus for exact coordination
        else:
            collusion_reward = max(0, 100 * (1 - price_diff_pct / 0.5))
        
        # Composite reward
        reward = (profit_reward * (1 - collusion_bonus_weight)) + (collusion_reward * collusion_bonus_weight)
        
        # Q-Learning update: Q(s,a) = Q(s,a) + lr * (reward + γ*maxQ(s',a') - Q(s,a))
        # Next state: assume competitor average moves slightly
        next_state_bin = np.argmin(np.abs(price_bins - competitor_avg * 1.01))
        max_next_q = np.max(q_tables[i][next_state_bin, :])
        
        old_q = q_tables[i][state_bin, action_bin]
        new_q = old_q + learning_rate * (reward + discount_factor * max_next_q - old_q)
        q_tables[i][state_bin, action_bin] = new_q
        
        new_prices[i] = candidate_price
        new_prices[i] = max(new_prices[i], base_costs[i] + 0.01)  # Cost floor
    
    prices = new_prices

print("\n[TRIAL 5] Creating visualization...")

fig = plt.figure(figsize=(14, 8), dpi=100)
fig.suptitle('Trial 5: Q-Learning with Collusion Incentive', fontsize=14, y=0.98)

days = list(range(len(data[airlines[0]]['prices'])))

# Plot 1: Price Trajectories
ax1 = plt.subplot(211)
for i, airline in enumerate(airlines):
    ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i], linewidth=2)

ax1.set_title("Price Trajectories (Learning to Coordinate)", fontsize=12, pad=6)
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

output_dir = "trials/trial5_qlearning_collusion"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/trial5_300dpi.png", dpi=300, bbox_inches='tight', facecolor='white')

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

print(f"[TRIAL 5] Saved to {output_dir}/")
print(f"[TRIAL 5] Export complete!\n")
