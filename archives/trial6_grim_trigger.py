"""
TRIAL 6: GRIM TRIGGER PUNISHMENT STRATEGY
Most sophisticated: Implements the classic game-theoretic model of collusion.
- Cooperate at agreed price (average of all market prices)
- Monitor for deviations (if any competitor deviates >threshold, enter punishment)
- Punishment: Drop price to near-cost for N days
- Recovery: Return to cooperation if no violations observed

This is the GOLD STANDARD for understanding collusion as a self-enforcing equilibrium.
Purpose: Show collusion when there is THREAT of retaliation.
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

# Grim trigger parameters
target_collusive_price = 125.0  # Agreed collusive price (cooperation)
deviation_threshold = 0.95  # If price < 95% of target, that's a deviation
punishment_duration = 10  # Days of price war if deviation detected
recovery_threshold = 5  # Need 5 days of peace to recover

print("[TRIAL 6] Grim Trigger Punishment (Game-Theoretic Collusion Model)")
print(f"  Duration: {num_days} days")
print(f"  Airlines: {num_airlines}")
print(f"  COLLUSIVE PRICE (cooperation): ${target_collusive_price}")
print(f"  DEVIATION THRESHOLD: {deviation_threshold*100:.0f}% of target (< ${target_collusive_price*deviation_threshold:.2f} triggers punishment)")
print(f"  PUNISHMENT PRICE: ${base_costs[0] + 1} (near-cost)")
print(f"  PUNISHMENT DURATION: {punishment_duration} days")
print(f"  RECOVERY THRESHOLD: {recovery_threshold} days of compliance")
print()

data = {airline: {'prices': [], 'profits': [], 'demands': [], 'market_shares': [], 'state': []} for airline in airlines}
prices = np.array([target_collusive_price] * num_airlines)  # Start at collusive price

# Track punishment state for each airline
punished_days = [0] * num_airlines  # Counts days in punishment
non_deviation_days = [0] * num_airlines  # Counts days without deviation
in_punishment = [False] * num_airlines  # Is airline currently being punished?

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
        
        # Record state for analysis
        if in_punishment[i]:
            data[airline]['state'].append('PUNISHED')
        else:
            data[airline]['state'].append('COOPERATING')
    
    # Check for deviations from all airlines
    for i in range(num_airlines):
        competitor_prices = np.delete(prices, i)
        competitor_avg = np.mean(competitor_prices)
        
        # Has someone deviated?
        min_competitor_price = np.min(competitor_prices)
        if min_competitor_price < target_collusive_price * deviation_threshold:
            # Deviation detected - enter punishment
            in_punishment[i] = True
            punished_days[i] = punishment_duration
            non_deviation_days[i] = 0
    
    # Update punishment state
    new_prices = np.zeros(num_airlines)
    
    for i in range(num_airlines):
        if in_punishment[i]:
            # In punishment: charge near-cost price
            new_prices[i] = base_costs[i] + 1.0
            punished_days[i] -= 1
            
            # Check if punishment period is over
            if punished_days[i] <= 0:
                in_punishment[i] = False
                non_deviation_days[i] = 0
        else:
            # Cooperating: charge collusive price
            # But tempted to deviate occasionally (simulate realistic oligopoly)
            if np.random.random() < 0.05:  # 5% chance to test undercutting
                new_prices[i] = target_collusive_price * 0.98  # Try to undercut slightly
            else:
                new_prices[i] = target_collusive_price
            
            non_deviation_days[i] += 1
    
    # Enforce cost floor
    new_prices = np.maximum(new_prices, base_costs + 0.01)
    prices = new_prices

print("\n[TRIAL 6] Creating visualization...")

fig = plt.figure(figsize=(14, 8), dpi=100)
fig.suptitle('Trial 6: Grim Trigger Punishment (Self-Enforcing Collusion)', fontsize=14, y=0.98)

days = list(range(len(data[airlines[0]]['prices'])))

# Plot 1: Price Trajectories
ax1 = plt.subplot(211)
for i, airline in enumerate(airlines):
    ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i], linewidth=2)

ax1.axhline(y=target_collusive_price, color='green', linestyle='--', linewidth=2, alpha=0.6, 
            label=f'Collusive Price (${target_collusive_price:.0f})')
ax1.axhline(y=base_costs[0]+1, color='red', linestyle='--', linewidth=1, alpha=0.5, 
            label=f'Punishment Price (${base_costs[0]+1:.0f})')
ax1.set_title("Price Trajectories (Cooperation vs. Punishment)", fontsize=12, pad=6)
ax1.set_ylabel("Price ($)", fontsize=11)
ax1.legend(loc='best', fontsize=8, framealpha=0.9, ncol=3)
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

output_dir = "trials/trial6_grim_trigger"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/trial6_300dpi.png", dpi=300, bbox_inches='tight', facecolor='white')

final_data = {
    "Airline": airlines,
    "Final_Price": [data[a]['prices'][-1] for a in airlines],
    "Final_Profit": [data[a]['profits'][-1] for a in airlines],
    "Avg_Price": [np.mean(data[a]['prices']) for a in airlines],
    "Price_Variance": [np.var(data[a]['prices']) for a in airlines],
    "Avg_Market_Share": [np.mean(data[a]['market_shares']) for a in airlines],
    "Days_Cooperating": [sum(1 for s in data[a]['state'] if s == 'COOPERATING') for a in airlines],
    "Days_Punished": [sum(1 for s in data[a]['state'] if s == 'PUNISHED') for a in airlines],
}
pd.DataFrame(final_data).to_csv(f"{output_dir}/final_stats.csv", index=False)

history_data = {"Day": list(range(num_days))}
for airline in airlines:
    history_data[f"{airline}_Price"] = data[airline]['prices']
    history_data[f"{airline}_State"] = data[airline]['state']

pd.DataFrame(history_data).to_csv(f"{output_dir}/full_history.csv", index=False)

print(f"[TRIAL 6] Saved to {output_dir}/")
print(f"[TRIAL 6] Export complete!\n")
