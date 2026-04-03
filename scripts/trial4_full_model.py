"""
TRIAL 4: FULL MODEL — Q-LEARNING + MIMICRY + MARKET SHARE GUARANTEE
The complete simulation combining all realistic market mechanisms:
  1. Q-learning agents (same as Trial 3)
  2. Mimicry: bottom-ranked airlines partially copy top performers
  3. Minimum 7% market share guarantee (regulatory protection)
  4. Collusion incentive in reward function

This represents a realistic oligopolistic market where:
  - Agents learn independently under incomplete information
  - Underperformers adapt by observing (not copying) leaders
  - Regulatory protections prevent total market capture
  - Coordination emerges from individual profit-seeking + stability preference

Expected: Tighter convergence than Trial 3, with prices stabilising
in an elevated band well above cost — tacit collusion.
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
min_market_share = 0.07  # 7% guarantee
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

# Q-learning params
learning_rate = 0.15
discount_factor = 0.95
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995

# State discretisation — RELATIVE price ratio (scale-invariant)
ratio_bins = [0.0, 0.85, 0.93, 0.98, 1.02, 1.07, 1.15, 999.0]  # 7 bins
n_ratio_bins = len(ratio_bins) - 1
profit_tiers = [float('-inf'), 0, 500, 2000, float('inf')]  # 4 bins
n_profit_bins = len(profit_tiers) - 1
rank_bins = 5
actions = np.array([0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10])
n_actions = len(actions)

collusion_band = 0.12
collusion_bonus = 0.5

mimicry_rate = 0.25
mimicry_threshold = 2

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

print("[TRIAL 4] Full Model — Q-Learning + Mimicry + Market Share Guarantee")
print(f"  Duration          : {num_days} days")
print(f"  Min market share  : {min_market_share*100:.0f}%")
print(f"  Mimicry rate      : {mimicry_rate*100:.0f}% toward leader")
print(f"  Mimicry threshold : bottom {mimicry_threshold} airlines")
print()

data = {a: {'prices': [], 'profits': [], 'market_shares': []} for a in airlines}

q_tables = [np.ones((n_states, n_actions)) * 50.0 for _ in range(num_airlines)]
prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
epsilon = epsilon_start

for day in range(num_days):
    if day % 250 == 0:
        print(f"  Day {day}/{num_days}  ε={epsilon:.3f}  avg price ${prices.mean():.1f}  spread ${prices.max()-prices.min():.1f}")

    # ── Market outcome with minimum share guarantee ──
    exp_u = np.exp(-alpha * prices)
    raw_shares = exp_u / exp_u.sum()

    # Enforce minimum market share
    shares = raw_shares.copy()
    below_min = shares < min_market_share
    if below_min.any():
        # Redistribute: give min to those below, take proportionally from those above
        deficit = np.sum(np.maximum(min_market_share - shares, 0))
        above_min = ~below_min
        if above_min.any():
            excess = shares[above_min] - min_market_share
            total_excess = excess.sum()
            if total_excess > 0:
                shares[above_min] -= deficit * (excess / total_excess)
            shares[below_min] = min_market_share

    demands = shares * market_demand_per_day
    profits = (prices - base_costs) * demands
    profit_ranks = np.argsort(-profits)  # index of best to worst

    for i, a in enumerate(airlines):
        data[a]['prices'].append(prices[i])
        data[a]['profits'].append(profits[i])
        data[a]['market_shares'].append(shares[i] * 100)

    # ── Q-learning update ──
    new_prices = np.zeros(num_airlines)
    for i in range(num_airlines):
        comp_prices = np.delete(prices, i)
        comp_avg = comp_prices.mean()
        my_rank = int(np.where(profit_ranks == i)[0][0])
        state = get_state(prices[i], comp_avg, my_rank, profits[i])
        s_idx = state_to_idx(state)

        if np.random.random() < epsilon:
            action_idx = np.random.randint(n_actions)
        else:
            action_idx = np.argmax(q_tables[i][s_idx])

        new_p = prices[i] * actions[action_idx]
        new_p = np.clip(new_p, price_floor, 200.0)

        # Reward
        test_prices = prices.copy()
        test_prices[i] = new_p
        eu = np.exp(-alpha * test_prices)
        sh = eu / eu.sum()
        # Apply min share to test too
        sh = np.maximum(sh, min_market_share)
        sh = sh / sh.sum()
        profit_reward = (new_p - base_costs[i]) * sh[i] * market_demand_per_day

        price_diff_pct = abs(new_p - comp_avg) / max(comp_avg, 1)
        if price_diff_pct <= collusion_band:
            price_above_cost = max(0, comp_avg - base_costs[i])
            premium = (price_above_cost / 40.0) ** 2
            closeness = 1.0 - price_diff_pct / collusion_band
            coord_reward = closeness * premium * 150
        else:
            coord_reward = 0.0

        reward = (1 - collusion_bonus) * profit_reward + collusion_bonus * coord_reward

        new_comp_avg = (comp_avg * (num_airlines - 1) + new_p) / num_airlines
        next_state = get_state(new_p, new_comp_avg, my_rank, profit_reward)
        ns_idx = state_to_idx(next_state)

        old_q = q_tables[i][s_idx, action_idx]
        max_next_q = np.max(q_tables[i][ns_idx])
        q_tables[i][s_idx, action_idx] = old_q + learning_rate * (
            reward + discount_factor * max_next_q - old_q
        )

        new_prices[i] = new_p

    # ── Mimicry: bottom-ranked airlines partially copy leader ──
    best_airline = profit_ranks[0]
    best_price = new_prices[best_airline]
    for rank_pos in range(num_airlines - mimicry_threshold, num_airlines):
        loser = profit_ranks[rank_pos]
        price_gap = best_price - new_prices[loser]
        if abs(price_gap) > 1.0:  # only mimic if meaningful gap
            new_prices[loser] += mimicry_rate * price_gap
            new_prices[loser] = np.clip(new_prices[loser], price_floor, 200.0)

    prices = new_prices
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

# ── Plotting ────────────────────────────────────────────────
print(f"\n[TRIAL 4] Final ε={epsilon:.3f}  avg price ${prices.mean():.1f}")
print("[TRIAL 4] Creating visualisation …")

fig, axes = plt.subplots(3, 1, figsize=(14, 11), dpi=600)
fig.suptitle('Trial 4: Full Model — Q-Learning + Mimicry + Market Share Guarantee', fontsize=14, y=0.99)
days_range = range(num_days)
window = 20

# Price trajectories
ax1 = axes[0]
for i, a in enumerate(airlines):
    ax1.plot(days_range, data[a]['prices'], label=a, color=colors[i], linewidth=1, alpha=0.7)
all_prices = np.array([data[a]['prices'] for a in airlines])
market_avg = all_prices.mean(axis=0)
if num_days > window:
    smoothed = np.convolve(market_avg, np.ones(window)/window, mode='valid')
    ax1.plot(range(window-1, num_days), smoothed, 'k--', lw=2.5, alpha=0.8, label='Market Avg (20d)')
ax1.axhline(y=65, color='black', ls=':', lw=1, alpha=.4, label='Cost ($65)')
ax1.set_ylabel("Price ($)")
ax1.set_title("Price Trajectories")
ax1.legend(loc='upper right', fontsize=7, ncol=4)
ax1.grid(True, alpha=.15)
ax1.set_xlim(0, num_days)

# Market share (smoothed)
ax2 = axes[1]
for i, a in enumerate(airlines):
    ms = np.array(data[a]['market_shares'])
    if num_days > window:
        ms_smooth = np.convolve(ms, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, num_days), ms_smooth, color=colors[i], linewidth=1.5)
ax2.axhline(y=20, color='gray', ls='--', lw=1, alpha=0.4, label='Equal share (20%)')
ax2.axhline(y=7, color='red', ls=':', lw=1, alpha=0.4, label='Min guarantee (7%)')
ax2.set_ylabel("Market Share (%) — 20d avg")
ax2.set_title("Market Share")
ax2.grid(True, alpha=.15)
ax2.set_xlim(0, num_days)
ax2.set_ylim(0, 50)
ax2.legend(loc='upper right', fontsize=8)

# Price spread
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
output_dir = "trials/trial4_full_model"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"trial4_600dpi.png", dpi=600, bbox_inches='tight', facecolor='white')

# Summary stats
final = {"Airline": airlines,
         "Final_Price": [data[a]['prices'][-1] for a in airlines],
         "Avg_Price_Last100": [np.mean(data[a]['prices'][-100:]) for a in airlines],
         "Avg_Share_Last100": [np.mean(data[a]['market_shares'][-100:]) for a in airlines]}
df = pd.DataFrame(final)
df.to_csv(f"{output_dir}/final_stats.csv", index=False)

# Collusion analysis
last_100_avg = market_avg[-100:]
last_100_spread = price_spread[-100:]
print(f"\n[TRIAL 4] === COLLUSION ANALYSIS (Last 100 days) ===")
print(f"  Avg market price : ${last_100_avg.mean():.2f}")
print(f"  Avg price spread : ${last_100_spread.mean():.2f}")
print(f"  Price above cost : ${last_100_avg.mean() - 65:.2f} ({(last_100_avg.mean()-65)/65*100:.1f}%)")
print(f"  Spread / Avg     : {last_100_spread.mean()/last_100_avg.mean()*100:.1f}%")
if last_100_spread.mean() / last_100_avg.mean() < 0.15 and last_100_avg.mean() > 90:
    print(f"  RESULT: Evidence of tacit collusion ✓")
else:
    print(f"  RESULT: Weak or no collusion")

print(f"\n[TRIAL 4] Saved to {output_dir}/\n")
