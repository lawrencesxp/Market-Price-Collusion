import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
import random
from collections import defaultdict
import pickle

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
num_days_pre_training = 500
num_days_multi_agent = 1000
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
base_costs = np.ones(num_airlines) * 65
alpha = 1.0  # Reduced to make price less dominant
market_demand_per_day = 1000
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
baseline_share = 0.05  # 5% baseline market share per airline

# Seasonality setup
seasonality_effect_pre = np.sin(np.linspace(0, 3 * np.pi, num_days_pre_training)) * 0.15
seasonality_effect_multi = np.sin(np.linspace(0, 3 * np.pi, num_days_multi_agent)) * 0.15

# Initialize data storage
data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': []} for airline in airlines}
current_day = 0
current_phase = "pre_training"
current_training_airline_idx = 0
is_running = False
pre_training_complete = False

# Initial prices
prices = np.array([100, 110, 120, 130, 140])

# Reinforcement Learning parameters
price_actions = [-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10]
learning_rate = 0.1
discount_factor = 0.9
exploration_rate_initial = 1.0
exploration_decay = 0.99
exploration_min = 0.01

# Initialize Q-tables and exploration rates
q_tables = {}
exploration_rates = {}
for airline in airlines:
    q_tables[airline] = defaultdict(lambda: np.zeros(len(price_actions)))
    exploration_rates[airline] = exploration_rate_initial

# Static competitor strategies
def get_static_price(day, strategy, base_price):
    if strategy == "fixed":
        return base_price
    elif strategy == "gradual_increase":
        return base_price * (1 + 0.001 * day)
    elif strategy == "gradual_decrease":
        return base_price * (1 - 0.001 * day)
    elif strategy == "high_price":
        return base_price * 1.2
    elif strategy == "low_price":
        return base_price * 0.8
    return base_price

# Progressive difficulty schedule
difficulty_schedule = [
    {"days": 20, "strategies": ["fixed"] * 4},
    {"days": 30, "strategies": ["fixed", "gradual_increase", "gradual_decrease", "fixed"]},
    {"days": 30, "strategies": ["high_price", "low_price", "gradual_increase", "gradual_decrease"]},
    {"days": 20, "strategies": ["high_price", "low_price", "fixed", "gradual_increase"]}
]

# Discretize state space
def get_state(airline_idx, prices, profits, rank):
    own_price = prices[airline_idx]
    price_tier = 0 if own_price < 80 else 1 if own_price < 120 else 2
    profit = profits[airline_idx]
    profit_tier = 0 if profit < 0 else 1 if profit < 10000 else 2
    avg_price = np.mean(prices)
    relative_price = 0 if own_price < avg_price * 0.9 else 2 if own_price > avg_price * 1.1 else 1
    return (price_tier, profit_tier, rank, relative_price)

# Select action
def select_action(airline, state):
    exploration_rate = exploration_rates[airline]
    if np.random.random() < exploration_rate:
        return np.random.randint(len(price_actions))
    return np.argmax(q_tables[airline][state])

# Update Q-table
def update_q_table(airline, state, action, reward, next_state):
    current_q = q_tables[airline][state][action]
    max_next_q = np.max(q_tables[airline][next_state])
    new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
    q_tables[airline][state][action] = new_q

# Save Q-table
def save_q_table(airline):
    with open(f"{airline}_q_table.pkl", "wb") as f:
        pickle.dump(dict(q_tables[airline]), f)
    print(f"Saved Q-table for {airline}")

# Load Q-table
def load_q_table(airline):
    try:
        with open(f"{airline}_q_table.pkl", "rb") as f:
            q_tables[airline] = defaultdict(lambda: np.zeros(len(price_actions)), pickle.load(f))
        print(f"Loaded Q-table for {airline}")
    except FileNotFoundError:
        print(f"No Q-table found for {airline}, initializing new Q-table")
        q_tables[airline] = defaultdict(lambda: np.zeros(len(price_actions)))

# Save PNG of current figure
def save_current_frame(filename):
    fig.savefig(filename)
    print(f"Saved plot as '{filename}'")

# GUI Setup
root = tk.Tk()
root.title("Airline Price Competition Simulation")

fig = Figure(figsize=(15, 10))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

control_frame = tk.Frame(root)
control_frame.pack(side=tk.BOTTOM, fill=tk.X)

speed_label = tk.Label(control_frame, text="Animation Speed (ms):")
speed_label.pack(side=tk.LEFT, padx=5)
speed_var = tk.IntVar(value= 2)
speed_slider = tk.Scale(control_frame, from_=10, to=500, orient=tk.HORIZONTAL, variable=speed_var)
speed_slider.pack(side=tk.LEFT, padx=5)

learn_var = tk.BooleanVar(value=True)
learn_check = tk.Checkbutton(control_frame, text="Enable Learning", variable=learn_var)
learn_check.pack(side=tk.LEFT, padx=10)

lr_label = tk.Label(control_frame, text="Learning Rate:")
lr_label.pack(side=tk.LEFT, padx=5)
lr_var = tk.DoubleVar(value=learning_rate)
lr_slider = tk.Scale(control_frame, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=lr_var)
lr_slider.pack(side=tk.LEFT, padx=5)

exploration_label = tk.Label(control_frame, text="Exploration Rate: 1.00")
exploration_label.pack(side=tk.LEFT, padx=10)

phase_label = tk.Label(control_frame, text="Phase: Pre-training")
phase_label.pack(side=tk.LEFT, padx=10)
training_airline_label = tk.Label(control_frame, text="Training: Airline_1")
training_airline_label.pack(side=tk.LEFT, padx=10)

info_frame = tk.Frame(root)
info_frame.pack(side=tk.BOTTOM, fill=tk.X)
day_label = tk.Label(info_frame, text="Day: 0")
day_label.pack(side=tk.LEFT, padx=5)
demand_label = tk.Label(info_frame, text="Total Market Demand: 1000")
demand_label.pack(side=tk.LEFT, padx=5)
avg_price_var = tk.StringVar(value="Avg. Market Price: $100.00")
avg_price_label = tk.Label(info_frame, textvariable=avg_price_var)
avg_price_label.pack(side=tk.LEFT, padx=5)

def reset_simulation():
    global current_day, prices, is_running, data, q_tables, exploration_rates, current_phase, current_training_airline_idx, pre_training_complete
    is_running = False
    start_button.config(text="Start Simulation")
    current_day = 0
    prices = np.array([100, 110, 120, 130, 140])
    data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': []} for airline in airlines}
    current_phase = "pre_training"
    current_training_airline_idx = 0
    pre_training_complete = False
    phase_label.config(text="Phase: Pre-training")
    training_airline_label.config(text=f"Training: {airlines[0]}")
    for airline in airlines:
        q_tables[airline] = defaultdict(lambda: np.zeros(len(price_actions)))
        exploration_rates[airline] = exploration_rate_initial
    exploration_label.config(text=f"Exploration Rate: {exploration_rate_initial:.2f}")
    day_label.config(text="Day: 0")
    demand_label.config(text="Total Market Demand: 1000")
    avg_price_var.set("Avg. Market Price: $100.00")
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    fig.tight_layout()
    canvas.draw()
    print("Simulation reset")

def simulate_day():
    global current_day, prices, is_running, exploration_rates, current_training_airline_idx, current_phase, pre_training_complete, data
    if not is_running:
        return

    learning_enabled = learn_var.get()
    learning_rate = lr_var.get()

    if current_phase == "pre_training":
        total_days = num_days_pre_training
        seasonality_effect = seasonality_effect_pre
        current_airline = airlines[current_training_airline_idx]
        day_in_phase = current_day % num_days_pre_training

        days_passed = day_in_phase
        cumulative_days = 0
        current_strategies = difficulty_schedule[0]["strategies"]
        for level in difficulty_schedule:
            if days_passed < cumulative_days + level["days"]:
                current_strategies = level["strategies"]
                break
            cumulative_days += level["days"]

        total_demand = market_demand_per_day * (1 + seasonality_effect[day_in_phase])
        avg_market_price = np.mean(prices)
        avg_price_var.set(f"Avg. Market Price: ${avg_market_price:.2f}")

        # Modified market share calculation
        baseline_demand = total_demand * baseline_share  # Guaranteed demand per airline
        price_dependent_demand = total_demand * (1 - baseline_share * num_airlines)  # Remaining demand
        exp_utils = np.exp(-alpha * prices)
        price_shares = exp_utils / np.sum(exp_utils)
        demand_shares = (baseline_share + (1 - baseline_share * num_airlines) * price_shares)
        demands = demand_shares * total_demand

        profits = np.zeros(num_airlines)
        for i, airline in enumerate(airlines):
            price = prices[i]
            demand = demands[i]
            cost = base_costs[i]
            profit = (price - cost) * demand
            data[airline]['prices'].append(price)
            data[airline]['profits'].append(profit)
            data[airline]['demands'].append(demand)
            data[airline]['cumulative_profits'].append(
                profit if len(data[airline]['cumulative_profits']) == 0 else
                data[airline]['cumulative_profits'][-1] + profit
            )
            profits[i] = profit

        profit_ranks = np.argsort(np.argsort(-profits))
        update_plots()

        new_prices = np.zeros(num_airlines)
        previous_states = {}
        actions_taken = {}

        for i, airline in enumerate(airlines):
            if airline == current_airline and learning_enabled:
                current_state = get_state(i, prices, profits, profit_ranks[i])
                previous_states[airline] = current_state
                action_idx = select_action(airline, current_state)
                actions_taken[airline] = action_idx
                price_change = price_actions[action_idx]
                new_price = prices[i] * (1 + price_change)
                new_price += np.random.normal(0, 2.0)
                new_price = max(new_price, base_costs[i] + 0.01)
                new_price = min(new_price, 180)
                new_prices[i] = new_price
                if exploration_rates[airline] > exploration_min:
                    exploration_rates[airline] *= exploration_decay
            else:
                base_price = [100, 110, 120, 130, 140][i]
                strategy = current_strategies[i % len(current_strategies)]
                new_price = get_static_price(day_in_phase, strategy, base_price)
                new_price = max(new_price, base_costs[i] + 0.01)
                new_price = min(new_price, 180)
                new_prices[i] = new_price

        avg_exploration = exploration_rates[current_airline]
        exploration_label.config(text=f"Exploration Rate: {avg_exploration:.2f}")

        prices = new_prices
        current_day += 1
        day_in_phase = current_day % num_days_pre_training

        if learning_enabled and current_day > 1:
            for i, airline in enumerate(airlines):
                if airline == current_airline:
                    reward = profits[i]
                    new_state = get_state(i, prices, profits, profit_ranks[i])
                    update_q_table(airline, previous_states[airline], actions_taken[airline], reward, new_state)

        if day_in_phase == 0 and current_day > 0:
            save_q_table(current_airline)
            save_current_frame(f"pre_training_final_{current_airline}.png")
            current_training_airline_idx += 1
            if current_training_airline_idx < num_airlines:
                training_airline_label.config(text=f"Training: {airlines[current_training_airline_idx]}")
                data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': []} for airline in airlines}
                prices = np.array([100, 110, 120, 130, 140])
                exploration_rates[current_airline] = exploration_rate_initial
                ax1.clear()
                ax2.clear()
                ax3.clear()
                ax4.clear()
                canvas.draw()
            else:
                current_phase = "multi_agent"
                pre_training_complete = True
                phase_label.config(text="Phase: Multi-agent")
                training_airline_label.config(text="Training: All Airlines")
                current_day = 0
                for airline in airlines:
                    load_q_table(airline)
                    exploration_rates[airline] = exploration_rate_initial
                data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': []} for airline in airlines}
                prices = np.array([100, 110, 120, 130, 140])
                ax1.clear()
                ax2.clear()
                ax3.clear()
                ax4.clear()
                canvas.draw()

    else:
        total_days = num_days_multi_agent
        seasonality_effect = seasonality_effect_multi
        if current_day >= total_days:
            is_running = False
            messagebox.showinfo("Simulation Complete", "The multi-agent simulation has reached the end!")
            save_current_frame("multi_agent_final.png")
            return

        total_demand = market_demand_per_day * (1 + seasonality_effect[current_day])
        avg_market_price = np.mean(prices)
        avg_price_var.set(f"Avg. Market Price: ${avg_market_price:.2f}")

        # Modified market share calculation
        baseline_demand = total_demand * baseline_share
        price_dependent_demand = total_demand * (1 - baseline_share * num_airlines)
        exp_utils = np.exp(-alpha * prices)
        price_shares = exp_utils / np.sum(exp_utils)
        demand_shares = (baseline_share + (1 - baseline_share * num_airlines) * price_shares)
        demands = demand_shares * total_demand

        profits = np.zeros(num_airlines)
        for i, airline in enumerate(airlines):
            price = prices[i]
            demand = demands[i]
            cost = base_costs[i]
            profit = (price - cost) * demand
            data[airline]['prices'].append(price)
            data[airline]['profits'].append(profit)
            data[airline]['demands'].append(demand)
            data[airline]['cumulative_profits'].append(
                profit if len(data[airline]['cumulative_profits']) == 0 else
                data[airline]['cumulative_profits'][-1] + profit
            )
            profits[i] = profit

        profit_ranks = np.argsort(np.argsort(-profits))
        update_plots()

        new_prices = np.zeros(num_airlines)
        previous_states = {}
        actions_taken = {}

        for i, airline in enumerate(airlines):
            current_state = get_state(i, prices, profits, profit_ranks[i])
            previous_states[airline] = current_state
            action_idx = select_action(airline, current_state)
            actions_taken[airline] = action_idx
            price_change = price_actions[action_idx]
            new_price = prices[i] * (1 + price_change)
            if learning_enabled:
                new_price += np.random.normal(0, 2.0)
            new_price = max(new_price, base_costs[i] + 0.01)
            new_price = min(new_price, 180)
            new_prices[i] = new_price
            if learning_enabled and exploration_rates[airline] > exploration_min:
                exploration_rates[airline] *= exploration_decay

        avg_exploration = np.mean([exploration_rates[airline] for airline in airlines])
        exploration_label.config(text=f"Exploration Rate: {avg_exploration:.2f}")

        if current_day % 15 == 0 and current_day > 0:
            shock_magnitude = np.random.uniform(0.85, 1.15)
            new_prices *= shock_magnitude
            print(f"Day {current_day}: Market disruption! Prices adjusted by {shock_magnitude:.2f}")
            messagebox.showinfo("Market Disruption",
                                f"Day {current_day}: Market disruption!\nPrices adjusted by {(shock_magnitude-1)*100:.1f}%.")

        prices = new_prices
        current_day += 1

        if learning_enabled and current_day > 1:
            for i, airline in enumerate(airlines):
                reward = profits[i]
                new_state = get_state(i, prices, profits, profit_ranks[i])
                update_q_table(airline, previous_states[airline], actions_taken[airline], reward, new_state)

    day_label.config(text=f"Day: {current_day}")
    demand_label.config(text=f"Total Market Demand: {total_demand:.2f}")

    if is_running:
        root.after(speed_var.get(), simulate_day)

def update_plots():
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    days = list(range(len(data[airlines[0]]['prices']))) if data[airlines[0]]['prices'] else [0]
    
    ax1.axhspan(65, 95, color='#DDFFDD', alpha=0.3)
    ax1.axhspan(95, 120, color='#FFFFDD', alpha=0.3)
    ax1.axhspan(120, 180, color='#FFDDDD', alpha=0.3)
    for i, airline in enumerate(airlines):
        ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i])
    ax1.set_title("Airline Price Trajectories")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, num_days_pre_training if current_phase == "pre_training" else num_days_multi_agent)
    ax1.set_ylim(60, 180)

    for i, airline in enumerate(airlines):
        ax2.plot(days, data[airline]['profits'], label=airline, color=colors[i])
    ax2.set_title("Airline Profit Trajectories")
    ax2.set_ylabel("Profit ($)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, num_days_pre_training if current_phase == "pre_training" else num_days_multi_agent)
    if days:
        max_profit = max([max(data[airline]['profits'] or [0]) for airline in airlines]) * 1.1
        min_profit = min([min(data[airline]['profits'] or [0]) for airline in airlines]) * 1.1
        if max_profit > 0:
            ax2.set_ylim(min_profit, max_profit)

    demand_data = np.array([data[airline]['demands'] for airline in airlines])
    ax3.stackplot(days, demand_data, labels=airlines, colors=colors, alpha=0.8)
    ax3.set_title("Customer Demand Share Over Time")
    ax3.set_ylabel("Demand")
    ax3.legend()
    ax3.grid(True)
    ax3.set_xlim(0, num_days_pre_training if current_phase == "pre_training" else num_days_multi_agent)
    ax3.set_ylim(0, market_demand_per_day * 1.2)

    for i, airline in enumerate(airlines):
        ax4.plot(days, data[airline]['cumulative_profits'], label=airline, color=colors[i])
    ax4.set_title("Cumulative Profits Over Time")
    ax4.set_xlabel("Day")
    ax4.set_ylabel("Cumulative Profit ($)")
    ax4.legend()
    ax4.grid(True)
    ax4.set_xlim(0, num_days_pre_training if current_phase == "pre_training" else num_days_multi_agent)
    if days:
        max_cum_profit = max([max(data[airline]['cumulative_profits'] or [0]) for airline in airlines])
        if max_cum_profit > 0:
            ax4.set_ylim(0, max_cum_profit * 1.1)

    fig.tight_layout()
    canvas.draw()

start_button = tk.Button(control_frame, text="Start Simulation", command=lambda: start_simulation())
start_button.pack(side=tk.LEFT, padx=5)
reset_button = tk.Button(control_frame, text="Reset Simulation", command=reset_simulation)
reset_button.pack(side=tk.LEFT, padx=5)

def start_simulation():
    global is_running
    if not is_running:
        is_running = True
        start_button.config(text="Pause Simulation")
        simulate_day()
    else:
        is_running = False
        start_button.config(text="Resume Simulation")

root.mainloop()