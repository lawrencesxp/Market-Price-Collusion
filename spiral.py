"""
TRIAL 0: PRICE WAR — Dramatic spiral below zero
Illustration only. 20 days, aggressive undercutting, no floor.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

num_days = 20
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

# Start at different prices
prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0], dtype=float)
price_history = [prices.copy()]

for day in range(num_days):
    new_prices = np.zeros(num_airlines)
    for i in range(num_airlines):
        # Each airline aggressively undercuts by 8-15% plus random noise
        cut = np.random.uniform(0.08, 0.15)
        noise = np.random.uniform(-5, 5)
        new_prices[i] = prices[i] * (1 - cut) + noise
    prices = new_prices
    price_history.append(prices.copy())

price_history = np.array(price_history)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
fig.set_facecolor('white')

for i in range(num_airlines):
    ax.plot(range(len(price_history)), price_history[:, i],
            color=colors[i], linewidth=2.5, label=airlines[i])

ax.axhline(y=0, color='black', ls='--', lw=1.5, alpha=0.5)
ax.set_xlim(0, num_days)
ax.set_ylim(-50, 160)
ax.set_xlabel("Time (days)", fontsize=12)
ax.set_ylabel("Price ($)", fontsize=12)
ax.set_title("Price War Without Bounds", fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig("spiral.png", dpi=600, bbox_inches='tight', facecolor='white')
print("Saved spiral.png")