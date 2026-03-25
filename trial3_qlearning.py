"""
TRIAL 3: Q-LEARNING WITH COLLUSION INCENTIVE
Airlines are Q-learning agents that learn pricing over 1000 days.
State: (own price tier, competitor avg tier, own market rank)
Action: price adjustment from discrete set
Reward: profit + bonus for staying within band of competitor average

Key design: high exploration early (ε decays from 1.0→0.05), giving agents
time to discover that coordination pays more than undercutting.

Expected: Early chaos → gradual convergence → emergent price coordination
above cost, WITHOUT explicit communication between agents.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# ── Parameters ──────────────────────────────────────────────
num_days = 1000
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
base_costs = np.ones(num_airlines) * 65
alpha = 0.08
market_demand_per_day = 1000
price_floor = 70.0
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

# Q-learning params
learning_rate = 0.15
discount_factor = 0.95
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995  # per day

# State discretisation — RELATIVE price ratio (scale-invariant)
# ratio = my_price / competitor_avg, binned into discrete levels
ratio_bins = [0.0, 0.85, 0.93, 0.98, 1.02, 1.07, 1.15, 999.0]  # 7 bins
n_ratio_bins = len(ratio_bins) - 1
# Profit tier: losing / low / medium / high
profit_tiers = [float('-inf'), 0, 500, 2000, float('inf')]  # 4 bins
n_profit_bins = len(profit_tiers) - 1
rank_bins = 5  # 1st through 5th

# Action space: price adjustments (multiplicative)
actions = np.array([0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10])
n_actions = len(actions)

# Collusion incentive
collusion_band = 0.12  # within 12% of competitor average
collusion_bonus = 0.5  # weight of coordination reward

def bin_ratio(ratio):
    for k in range(n_ratio_bins):
        if ratio < ratio_bins[k + 1]:
            return k
    return n_ratio_bins - 1

def bin_profit(profit):
    for k in range(n_profit_bins):
        if profit < profit_tiers[k + 1]:
            return k
    return n_profit_bins - 1

def get_state(my_price, comp_avg, my_rank, my_profit):
    ratio = my_price / max(comp_avg, 1)
    rb = bin_ratio(ratio)
    pb = bin_profit(my_profit)
    rk = min(my_rank, rank_bins - 1)
    return (rb, pb, rk)

n_states = n_ratio_bins * n_profit_bins * rank_bins

def state_to_idx(s):
    return s[0] * n_profit_bins * rank_bins + s[1] * rank_bins + s[2]

print("[TRIAL 3] Q-Learning with Collusion Incentive")
print(f"  Duration       : {num_days} days")
print(f"  State space    : {n_states} states (ratio × profit × rank)")
print(f"  Actions        : {n_actions}")
print(f"  Epsilon decay  : {epsilon_start} → {epsilon_end}")
print(f"  Collusion band : ±{collusion_band*100:.0f}% of competitor avg")
print(f"  Collusion weight: {collusion_bonus*100:.0f}%")
print()

data = {a: {'prices': [], 'profits': [], 'market_shares': []} for a in airlines}

# Initialise Q-tables with small optimistic values to encourage exploration
q_tables = [np.ones((n_states, n_actions)) * 50.0 for _ in range(num_airlines)]
prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
epsilon = epsilon_start

for day in range(num_days):
    if day % 250 == 0:
        print(f"  Day {day}/{num_days}  ε={epsilon:.3f}  avg price ${prices.mean():.1f}  spread ${prices.max()-prices.min():.1f}")

    # ── Market outcome ──
    exp_u = np.exp(-alpha * prices)
    shares = exp_u / exp_u.sum()
    demands = shares * market_demand_per_day
    profits = (prices - base_costs) * demands
    ranks = np.argsort(-profits)  # 0 = best

    for i, a in enumerate(airlines):
        data[a]['prices'].append(prices[i])
        data[a]['profits'].append(profits[i])
        data[a]['market_shares'].append(shares[i] * 100)

    # ── Q-learning update for each airline ──
    new_prices = np.zeros(num_airlines)
    for i in range(num_airlines):
        comp_prices = np.delete(prices, i)
        comp_avg = comp_prices.mean()
        my_rank = int(np.where(ranks == i)[0][0])
        state = get_state(prices[i], comp_avg, my_rank, profits[i])
        s_idx = state_to_idx(state)

        # ε-greedy action selection
        if np.random.random() < epsilon:
            action_idx = np.random.randint(n_actions)
        else:
            action_idx = np.argmax(q_tables[i][s_idx])

        # Apply action
        new_p = prices[i] * actions[action_idx]
        new_p = np.clip(new_p, price_floor, 200.0)

        # ── Compute reward ──
        test_prices = prices.copy()
        test_prices[i] = new_p
        eu = np.exp(-alpha * test_prices)
        sh = eu / eu.sum()
        profit_reward = (new_p - base_costs[i]) * sh[i] * market_demand_per_day

        # Collusion component: reward for being near competitor average
        # Key insight: reward scales with price level squared, so agents discover
        # that coordinating at HIGH prices is far more valuable than at low prices
        price_diff_pct = abs(new_p - comp_avg) / max(comp_avg, 1)
        if price_diff_pct <= collusion_band:
            # Premium scales quadratically with how far above cost the average is
            price_above_cost = max(0, comp_avg - base_costs[i])
            premium = (price_above_cost / 40.0) ** 2  # quadratic scaling
            closeness = 1.0 - price_diff_pct / collusion_band
            coord_reward = closeness * premium * 150
        else:
            coord_reward = 0.0

        reward = (1 - collusion_bonus) * profit_reward + collusion_bonus * coord_reward

        # ── Next state (estimate) ──
        new_comp_avg = (comp_avg * (num_airlines - 1) + new_p) / num_airlines
        new_rank = my_rank
        next_state = get_state(new_p, new_comp_avg, new_rank, profit_reward)
        ns_idx = state_to_idx(next_state)

        # Q-update
        old_q = q_tables[i][s_idx, action_idx]
        max_next_q = np.max(q_tables[i][ns_idx])
        q_tables[i][s_idx, action_idx] = old_q + learning_rate * (
            reward + discount_factor * max_next_q - old_q
        )

        new_prices[i] = new_p

    prices = new_prices
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

# ── Plotting ────────────────────────────────────────────────
print(f"\n[TRIAL 3] Final ε={epsilon:.3f}  avg price ${prices.mean():.1f}")
print("[TRIAL 3] Creating visualisation …")

fig, axes = plt.subplots(3, 1, figsize=(14, 11), dpi=100)
fig.suptitle('Trial 3: Q-Learning with Collusion Incentive (Emergent Coordination)', fontsize=14, y=0.99)
days_range = range(num_days)

# Price trajectories
ax1 = axes[0]
for i, a in enumerate(airlines):
    ax1.plot(days_range, data[a]['prices'], label=a, color=colors[i], linewidth=1, alpha=0.8)
# Rolling average
all_prices = np.array([data[a]['prices'] for a in airlines])
market_avg = all_prices.mean(axis=0)
window = 20
if num_days > window:
    smoothed = np.convolve(market_avg, np.ones(window)/window, mode='valid')
    ax1.plot(range(window-1, num_days), smoothed, 'k--', lw=2, alpha=0.7, label='Market Avg (20d)')
ax1.axhline(y=65, color='black', ls=':', lw=1, alpha=.4, label='Cost ($65)')
ax1.set_ylabel("Price ($)")
ax1.set_title("Price Trajectories")
ax1.legend(loc='upper right', fontsize=7, ncol=4)
ax1.grid(True, alpha=.15)
ax1.set_xlim(0, num_days)

# Market share
ax2 = axes[1]
for i, a in enumerate(airlines):
    # Smooth market share for readability
    ms = np.array(data[a]['market_shares'])
    if num_days > window:
        ms_smooth = np.convolve(ms, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, num_days), ms_smooth, color=colors[i], linewidth=1.5)
    else:
        ax2.plot(days_range, ms, color=colors[i], linewidth=1.5)
ax2.axhline(y=20, color='gray', ls='--', lw=1, alpha=0.4, label='Equal share (20%)')
ax2.set_ylabel("Market Share (%) — 20d avg")
ax2.set_title("Market Share")
ax2.grid(True, alpha=.15)
ax2.set_xlim(0, num_days)
ax2.set_ylim(0, 50)
ax2.legend(loc='upper right', fontsize=8)

# Price spread (convergence metric)
ax3 = axes[2]
price_spread = all_prices.max(axis=0) - all_prices.min(axis=0)
if num_days > window:
    spread_smooth = np.convolve(price_spread, np.ones(window)/window, mode='valid')
    ax3.plot(range(window-1, num_days), spread_smooth, color='#2C3E50', linewidth=2)
ax3.set_ylabel("Price Spread ($)")
ax3.set_xlabel("Day")
ax3.set_title("Price Spread (Max − Min) — Convergence Indicator")
ax3.grid(True, alpha=.15)
ax3.set_xlim(0, num_days)

plt.tight_layout()
output_dir = "trials/trial3_qlearning"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/trial3_300dpi.png", dpi=300, bbox_inches='tight', facecolor='white')

final = {"Airline": airlines,
         "Final_Price": [data[a]['prices'][-1] for a in airlines],
         "Avg_Price_Last100": [np.mean(data[a]['prices'][-100:]) for a in airlines],
         "Avg_Share_Last100": [np.mean(data[a]['market_shares'][-100:]) for a in airlines]}
pd.DataFrame(final).to_csv(f"{output_dir}/final_stats.csv", index=False)
print(f"[TRIAL 3] Saved to {output_dir}/\n")
