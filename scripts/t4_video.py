"""
TRIAL 4: FULL MODEL — Q-LEARNING + MIMICRY + MARKET SHARE GUARANTEE
Generates a high-quality MP4 animation of the simulation running day by day.
Also prints average price per airline over the last 100 days.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
min_market_share = 0.07
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

# Q-learning params
learning_rate = 0.15
discount_factor = 0.95
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995

# State discretisation — RELATIVE price ratio
ratio_bins = [0.0, 0.85, 0.93, 0.98, 1.02, 1.07, 1.15, 999.0]
n_ratio_bins = len(ratio_bins) - 1
profit_tiers = [float('-inf'), 0, 500, 2000, float('inf')]
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

# ══════════════════════════════════════════════════════════════
# RUN SIMULATION (collect all data first)
# ══════════════════════════════════════════════════════════════
print("[TRIAL 4] Running simulation...")

data = {a: {'prices': [], 'profits': [], 'market_shares': []} for a in airlines}
q_tables = [np.ones((n_states, n_actions)) * 50.0 for _ in range(num_airlines)]
prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
epsilon = epsilon_start
epsilon_history = []

for day in range(num_days):
    if day % 250 == 0:
        print(f"  Day {day}/{num_days}  ε={epsilon:.3f}  avg price ${prices.mean():.1f}")

    epsilon_history.append(epsilon)

    exp_u = np.exp(-alpha * prices)
    raw_shares = exp_u / exp_u.sum()
    shares = raw_shares.copy()
    below_min = shares < min_market_share
    if below_min.any():
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
    profit_ranks = np.argsort(-profits)

    for i, a in enumerate(airlines):
        data[a]['prices'].append(prices[i])
        data[a]['profits'].append(profits[i])
        data[a]['market_shares'].append(shares[i] * 100)

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

        test_prices = prices.copy()
        test_prices[i] = new_p
        eu = np.exp(-alpha * test_prices)
        sh = eu / eu.sum()
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

    best_airline = profit_ranks[0]
    best_price = new_prices[best_airline]
    for rank_pos in range(num_airlines - mimicry_threshold, num_airlines):
        loser = profit_ranks[rank_pos]
        price_gap = best_price - new_prices[loser]
        if abs(price_gap) > 1.0:
            new_prices[loser] += mimicry_rate * price_gap
            new_prices[loser] = np.clip(new_prices[loser], price_floor, 200.0)

    prices = new_prices
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

print("[TRIAL 4] Simulation complete.\n")

# ══════════════════════════════════════════════════════════════
# PRINT FINAL STATS
# ══════════════════════════════════════════════════════════════
all_prices = np.array([data[a]['prices'] for a in airlines])
market_avg = all_prices.mean(axis=0)
price_spread = all_prices.max(axis=0) - all_prices.min(axis=0)

print("=" * 60)
print("FINAL RESULTS — Average Price Per Airline (Last 100 Days)")
print("=" * 60)
for a in airlines:
    avg_last_100 = np.mean(data[a]['prices'][-100:])
    print(f"  {a}: ${avg_last_100:.2f}")
print(f"\n  Market Average (Last 100 Days): ${market_avg[-100:].mean():.2f}")
print(f"  Average Spread (Last 100 Days): ${price_spread[-100:].mean():.2f}")
print(f"  Price Above Cost: ${market_avg[-100:].mean() - 65:.2f} ({(market_avg[-100:].mean()-65)/65*100:.1f}%)")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# CREATE ANIMATED VIDEO
# ══════════════════════════════════════════════════════════════
print("\n[TRIAL 4] Creating animation (this may take a few minutes)...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=150)
fig.set_facecolor('white')

# Fixed axis limits for stable video
ax1.set_xlim(0, num_days)
ax1.set_ylim(50, 150)
ax1.set_ylabel("Price ($)", fontsize=12)
ax1.set_xlabel("Time (days)", fontsize=12)
ax1.set_title("Price Trajectories", fontsize=13, pad=8)
ax1.axhline(y=70, color='black', ls=':', lw=1, alpha=0.4, label='Cost ($70)')
ax1.grid(True, alpha=0.15)

ax2.set_xlim(0, num_days)
ax2.set_ylim(0, 50)
ax2.set_ylabel("Market Share (%)", fontsize=12)
ax2.set_xlabel("Day", fontsize=12)
ax2.set_title("Market Share", fontsize=13, pad=8)
ax2.axhline(y=20, color='gray', ls='--', lw=1, alpha=0.4)
ax2.axhline(y=7, color='red', ls=':', lw=1, alpha=0.4)
ax2.grid(True, alpha=0.15)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Line objects for each airline
price_lines = []
share_lines = []
for i, a in enumerate(airlines):
    pl, = ax1.plot([], [], color=colors[i], linewidth=1.5, alpha=0.8, label=a)
    price_lines.append(pl)
    sl, = ax2.plot([], [], color=colors[i], linewidth=1.5, alpha=0.8)
    share_lines.append(sl)

# Market average line
avg_line, = ax1.plot([], [], 'k--', linewidth=2.5, alpha=0.7, label='Market Avg')

# Day counter and epsilon display
day_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12,
                    fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax1.legend(loc='upper right', fontsize=8, ncol=4)

# We animate every 2nd day to keep video length reasonable (500 frames)
frame_step = 1
total_frames = num_days // frame_step

def init():
    for pl in price_lines:
        pl.set_data([], [])
    for sl in share_lines:
        sl.set_data([], [])
    avg_line.set_data([], [])
    day_text.set_text('')
    return price_lines + share_lines + [avg_line, day_text]

def animate(frame):
    day = frame * frame_step
    days_so_far = list(range(day + 1))

    for i, a in enumerate(airlines):
        price_lines[i].set_data(days_so_far, data[a]['prices'][:day + 1])
        share_lines[i].set_data(days_so_far, data[a]['market_shares'][:day + 1])

    avg_line.set_data(days_so_far, market_avg[:day + 1])

    eps = epsilon_history[min(day, len(epsilon_history) - 1)]
    spread = price_spread[day] if day < len(price_spread) else 0
    day_text.set_text(f'Day {day}  |  ε = {eps:.3f}  |  Spread: ${spread:.1f}  |  Avg: ${market_avg[day]:.1f}')

    return price_lines + share_lines + [avg_line, day_text]

anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=total_frames, interval=40, blit=True)

# Save as high-quality MP4
# Requires ffmpeg installed. If not available, falls back to pillow gif.
output_filename = "trial4_animation.mp4"
try:
    writer = animation.FFMpegWriter(fps=30, bitrate=5000,
                                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    anim.save(output_filename, writer=writer)
    print(f"[TRIAL 4] Animation saved as {output_filename} (MP4, 30fps)")
except Exception as e:
    print(f"[TRIAL 4] FFmpeg not available ({e}), trying GIF fallback...")
    output_filename = "trial4_animation.gif"
    try:
        writer = animation.PillowWriter(fps=25)
        anim.save(output_filename, writer=writer)
        print(f"[TRIAL 4] Animation saved as {output_filename} (GIF, 25fps)")
    except Exception as e2:
        print(f"[TRIAL 4] Could not save animation: {e2}")
        print("[TRIAL 4] Install ffmpeg with: pip install ffmpeg-python  OR  brew install ffmpeg / apt install ffmpeg")

plt.close()
print("[TRIAL 4] Done.\n")