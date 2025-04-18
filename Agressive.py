import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure

# Set random seed for reproducibility
np.random.seed(17)

# Configuration
num_days = 100
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
base_costs = np.random.uniform(50, 80, size=num_airlines)  # Marginal costs per airline
alpha = 0.5  # Price sensitivity for logit model
market_demand_per_day = 1000
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']  # Colors for each airline

# Seasonality setup
seasonality_effect = np.sin(np.linspace(0, 3 * np.pi, num_days)) * 0.1  # ±10% effect

# Initialize data storage
data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': []} for airline in airlines}
current_day = 0
is_running = False
price_history_window = 5  # Number of days to check for equilibrium
price_change_threshold = 0.5  # Max price change to consider equilibrium (in $)
last_place_days = {airline: 0 for airline in airlines}  # Track days in last place
last_place_threshold = 20  # Number of days before drastic action

# Initialize prices for the first day
prices = base_costs + 20  # Starting point: cost + $20
last_few_prices = [[] for _ in range(num_airlines)]  # Store recent prices for equilibrium check

# GUI Setup
root = tk.Tk()
root.title("Airline Price Competition Simulation")

# Create a figure with four subplots
fig = Figure(figsize=(15, 10))
ax1 = fig.add_subplot(411)  # Prices
ax2 = fig.add_subplot(412)  # Profits
ax3 = fig.add_subplot(413)  # Demand shares
ax4 = fig.add_subplot(414)  # Cumulative profits

# Embed the figure in the tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Control frame for buttons and sliders
control_frame = tk.Frame(root)
control_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Speed control slider
speed_label = tk.Label(control_frame, text="Animation Speed (ms):")
speed_label.pack(side=tk.LEFT, padx=5)
speed_var = tk.IntVar(value=100)  # Default speed: 100ms
speed_slider = tk.Scale(control_frame, from_=10, to=500, orient=tk.HORIZONTAL, variable=speed_var)
speed_slider.pack(side=tk.LEFT, padx=5)

# Current day and market demand display
info_frame = tk.Frame(root)
info_frame.pack(side=tk.BOTTOM, fill=tk.X)
day_label = tk.Label(info_frame, text="Day: 0")
day_label.pack(side=tk.LEFT, padx=5)
demand_label = tk.Label(info_frame, text="Total Market Demand: 1000")
demand_label.pack(side=tk.LEFT, padx=5)

# Function to check for equilibrium
def check_equilibrium():
    if current_day < price_history_window:
        return False

    for i in range(num_airlines):
        recent_prices = last_few_prices[i][-price_history_window:]
        if len(recent_prices) < price_history_window:
            return False
        price_changes = np.abs(np.diff(recent_prices))
        if np.any(price_changes > price_change_threshold):
            return False
    return True

# Function to display equilibrium message
def show_equilibrium_message():
    # Gather final state data
    final_prices = {airline: data[airline]['prices'][-1] for airline in airlines}
    final_profits = {airline: data[airline]['profits'][-1] for airline in airlines}
    final_demands = {airline: data[airline]['demands'][-1] for airline in airlines}
    final_cumulative_profits = {airline: data[airline]['cumulative_profits'][-1] for airline in airlines}

    # Construct the message
    message = "Equilibrium Reached!\n\n"
    message += f"The simulation has reached an equilibrium state on Day {current_day}.\n"
    message += "Reason: Prices have stabilized with minimal changes over the last 5 days.\n\n"
    message += "Final State:\n"
    for airline in airlines:
        message += f"{airline}:\n"
        message += f"  Price: ${final_prices[airline]:.2f}\n"
        message += f"  Profit: ${final_profits[airline]:.2f}\n"
        message += f"  Demand: {final_demands[airline]:.2f} customers\n"
        message += f"  Cumulative Profit: ${final_cumulative_profits[airline]:.2f}\n\n"

    # Display pop-up
    messagebox.showinfo("Equilibrium Reached", message)

    # Save final state to CSV
    final_data = {
        "Airline": airlines,
        "Final_Price": [final_prices[a] for a in airlines],
        "Final_Profit": [final_profits[a] for a in airlines],
        "Final_Demand": [final_demands[a] for a in airlines],
        "Cumulative_Profit": [final_cumulative_profits[a] for a in airlines]
    }
    df = pd.DataFrame(final_data)
    df.to_csv("final_state.csv", index=False)

# Function to simulate one day and update the plots
def simulate_day():
    global current_day, prices, is_running

    if not is_running or current_day >= num_days:
        return

    # Seasonality multiplier for the day
    seasonality_multiplier = 1 + seasonality_effect[current_day]
    total_demand = market_demand_per_day * seasonality_multiplier

    # Logit demand share model
    exp_util = np.exp(-alpha * prices)
    demand_shares = exp_util / np.sum(exp_util)

    # Compute daily demand
    demands = demand_shares * total_demand

    # Compute profits and store data
    profits = np.zeros(num_airlines)
    for i, airline in enumerate(airlines):
        price = prices[i]
        demand = demands[i]
        cost = base_costs[i]
        profit = (price - cost) * demand

        # Store data
        data[airline]['prices'].append(price)
        data[airline]['profits'].append(profit)
        data[airline]['demands'].append(demand)
        # Update cumulative profits
        if len(data[airline]['cumulative_profits']) == 0:
            data[airline]['cumulative_profits'].append(profit)
        else:
            data[airline]['cumulative_profits'].append(data[airline]['cumulative_profits'][-1] + profit)
        profits[i] = profit

        # Store recent prices for equilibrium check
        last_few_prices[i].append(price)

    # Update plots
    days = list(range(current_day + 1))

    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # Plot 1: Price Trajectories
    for i, airline in enumerate(airlines):
        ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i])
    ax1.set_title("Airline Price Trajectories")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, num_days)
    # Remove hardcoded y-limits to accommodate potential negative prices
    # ax1.set_ylim(50, 120)

    # Plot 2: Profit Trajectories
    for i, airline in enumerate(airlines):
        ax2.plot(days, data[airline]['profits'], label=airline, color=colors[i])
    ax2.set_title("Airline Profit Trajectories")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Profit ($)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, num_days)
    # Remove hardcoded y-limits to accommodate larger profit swings
    # ax2.set_ylim(-5000, 15000)

    # Plot 3: Demand Shares (Stacked Area)
    demand_data = np.array([data[airline]['demands'] for airline in airlines])
    ax3.stackplot(days, demand_data, labels=airlines, colors=colors, alpha=0.8)
    ax3.set_title("Customer Demand Share Over Time")
    ax3.set_xlabel("Day")
    ax3.set_ylabel("Demand (Customers)")
    ax3.legend()
    ax3.grid(True)
    ax3.set_xlim(0, num_days)
    ax3.set_ylim(0, market_demand_per_day * 1.2)

    # Plot 4: Cumulative Profits
    for i, airline in enumerate(airlines):
        ax4.plot(days, data[airline]['cumulative_profits'], label=airline, color=colors[i])
    ax4.set_title("Cumulative Profits Over Time")
    ax4.set_xlabel("Day")
    ax4.set_ylabel("Cumulative Profit ($)")
    ax4.legend()
    ax4.grid(True)
    ax4.set_xlim(0, num_days)
    # Set y-limit dynamically based on data
    max_cum_profit = max([max(data[airline]['cumulative_profits']) for airline in airlines])
    ax4.set_ylim(0, max_cum_profit * 1.1)

    # Adjust layout and redraw
    fig.tight_layout()
    canvas.draw()

    # Update day and demand labels
    day_label.config(text=f"Day: {current_day}")
    demand_label.config(text=f"Total Market Demand: {total_demand:.2f}")

    # Check for equilibrium
    if check_equilibrium():
        is_running = False
        show_equilibrium_message()
        return

    # Initialize ranks with a default value (neutral ranking) on day 0
    if current_day == 0:
        # Default ranking based on initial prices (higher price = lower rank)
        ranks = np.argsort(prices)[::-1]  # Higher price gets lower rank initially
        last_place_airline = airlines[ranks[-1]]
    else:
        # Rank airlines based on cumulative profits
        cum_profits = [data[airline]['cumulative_profits'][-1] for airline in airlines]
        # Rank from highest to lowest profit (rank 1 = highest profit)
        ranks = np.argsort(cum_profits)[::-1]  # Indices of sorted profits, descending
        last_place_idx = ranks[-1]  # Index of the airline with the lowest profit
        last_place_airline = airlines[last_place_idx]

        # Update last place counter
        for airline in airlines:
            if airline == last_place_airline:
                last_place_days[airline] += 1
            else:
                last_place_days[airline] = 0  # Reset if not in last place

    # Adjust prices for the next day
    new_prices = np.zeros(num_airlines)
    avg_price = np.mean(prices)
    for i, airline in enumerate(airlines):
        profit = profits[i]
        demand = demands[i]
        own_price = prices[i]
        rival_prices = np.delete(prices, i)
        avg_rival_price = np.mean(rival_prices)

        # Compute momentum: average price change over the last few days
        momentum = 0
        if len(last_few_prices[i]) >= price_history_window:
            recent_prices = last_few_prices[i][-price_history_window:]
            price_changes = np.diff(recent_prices)
            momentum = np.mean(price_changes) * 0.5  # Scale momentum effect

        # Check if this airline has been in last place for over 20 days
        drastic_action = False
        if current_day > last_place_threshold and last_place_days[airline] >= last_place_threshold:
            drastic_action = True
            new_prices[i] = own_price * 0.80  # More aggressive: 20% price reduction (was 15%)
        else:
            # More aggressive heuristic price adjustments
            if profit > 5000:  # High profit: increase price more aggressively
                new_prices[i] = own_price * 1.10  # 10% increase (was 5%)
            elif demand < 150:  # Low demand: decrease price more aggressively
                new_prices[i] = own_price * 0.90  # 10% decrease (was 5%)
            elif own_price > avg_rival_price * 1.05:  # Lower price if too expensive
                new_prices[i] = own_price * 0.95  # 5% decrease (was 2%)
            elif own_price < avg_rival_price * 0.95:  # Raise price if too cheap
                new_prices[i] = own_price * 1.05  # 5% increase (was 2%)
            else:
                new_prices[i] = own_price

            # Add momentum to encourage continued movement
            new_prices[i] += momentum

        # Rank-based noise adjustment
        # Find the airline's rank (0 = 1st place, 4 = 5th place)
        airline_rank = np.where(ranks == i)[0][0]
        if airline_rank == 0:  # Top-ranked airline gets higher negative noise
            noise = np.random.normal(-5, 3)  # Mean -5, std 3
        elif airline_rank == 4:  # Last-ranked airline gets higher positive noise
            noise = np.random.normal(5, 3)  # Mean +5, std 3
        else:  # Middle ranks get standard noise
            noise = np.random.normal(0, 3)

        new_prices[i] += noise

        # Removed price clipping to allow unrestricted price movement
        # new_prices[i] = np.clip(new_prices[i], base_costs[i] + 5, base_costs[i] + 40)

        # Notify if drastic action was taken
        if drastic_action:
            print(f"Day {current_day}: {airline} has been in last place for {last_place_days[airline]} days. Drastically reducing price by 20% to ${new_prices[i]:.2f}")

    prices = new_prices
    current_day += 1

    # Schedule the next day
    if current_day < num_days and is_running:
        root.after(speed_var.get(), simulate_day)

# Start/Pause/Resume functionality
def start_simulation():
    global is_running
    if not is_running:
        is_running = True
        start_button.config(text="Pause Simulation")
        simulate_day()
    else:
        is_running = False
        start_button.config(text="Resume Simulation")

# Start Button
start_button = tk.Button(control_frame, text="Start Simulation", command=start_simulation)
start_button.pack(side=tk.LEFT, padx=5)

# Run the GUI
root.mainloop()

# Save static plots at the end
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

days = list(range(len(data[airlines[0]]['prices'])))

# Plot 1: Price Trajectories
for i, airline in enumerate(airlines):
    ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i])
ax1.set_title("Airline Price Trajectories")
ax1.set_xlabel("Day")
ax1.set_ylabel("Price ($)")
ax1.legend()
ax1.grid(True)

# Plot 2: Profit Trajectories
for i, airline in enumerate(airlines):
    ax2.plot(days, data[airline]['profits'], label=airline, color=colors[i])
ax2.set_title("Airline Profit Trajectories")
ax2.set_xlabel("Day")
ax2.set_ylabel("Profit ($)")
ax2.legend()
ax2.grid(True)

# Plot 3: Demand Shares
demand_data = np.array([data[airline]['demands'] for airline in airlines])
ax3.stackplot(days, demand_data, labels=airlines, colors=colors, alpha=0.8)
ax3.set_title("Customer Demand Share Over Time")
ax3.set_xlabel("Day")
ax3.set_ylabel("Demand (Customers)")
ax3.legend()
ax3.grid(True)

# Plot 4: Cumulative Profits
for i, airline in enumerate(airlines):
    ax4.plot(days, data[airline]['cumulative_profits'], label=airline, color=colors[i])
ax4.set_title("Cumulative Profits Over Time")
ax4.set_xlabel("Day")
ax4.set_ylabel("Cumulative Profit ($)")
ax4.legend()
ax4.grid(True)

fig.tight_layout()
plt.savefig('agressive.png')
print("Static plot saved as 'agressive.png'.")