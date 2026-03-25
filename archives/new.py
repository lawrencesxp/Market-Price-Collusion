import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
num_days = 100
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
# Base costs (minimum amount to make profit is $65)
base_costs = np.ones(num_airlines) * 65
# High price sensitivity - customers strongly prefer cheaper airlines
alpha = 1.5  # Single high value for strong price sensitivity
market_demand_per_day = 1000
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']  # Colors for each airline

# Seasonality setup
seasonality_effect = np.sin(np.linspace(0, 3 * np.pi, num_days)) * 0.15

# Initialize data storage
data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': []} for airline in airlines}
current_day = 0
is_running = False
price_history_window = 10
price_change_threshold = 1.0
last_place_days = {airline: 0 for airline in airlines}
last_place_threshold = 5  # Take drastic action sooner (after just 5 days in last place)

# Initialize prices for the first day (as requested: 100, 110, 120, 130, 140)
prices = np.array([100, 110, 120, 130, 140])
last_few_prices = [[] for _ in range(num_airlines)]

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
speed_var = tk.IntVar(value=100)
speed_slider = tk.Scale(control_frame, from_=10, to=500, orient=tk.HORIZONTAL, variable=speed_var)
speed_slider.pack(side=tk.LEFT, padx=5)

# Current day and market demand display
info_frame = tk.Frame(root)
info_frame.pack(side=tk.BOTTOM, fill=tk.X)
day_label = tk.Label(info_frame, text="Day: 0")
day_label.pack(side=tk.LEFT, padx=5)
demand_label = tk.Label(info_frame, text="Total Market Demand: 1000")
demand_label.pack(side=tk.LEFT, padx=5)
avg_price_var = tk.StringVar(value="Avg. Market Price: $100.00")
avg_price_label = tk.Label(info_frame, textvariable=avg_price_var)
avg_price_label.pack(side=tk.LEFT, padx=5)

# Reset functionality
def reset_simulation():
    global current_day, prices, is_running, data, last_place_days, last_few_prices
    
    # Stop the simulation if running
    is_running = False
    start_button.config(text="Start Simulation")
    
    # Reset all variables
    current_day = 0
    prices = np.array([100, 110, 120, 130, 140])  # Reset to initial prices
    data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': []} for airline in airlines}
    last_place_days = {airline: 0 for airline in airlines}
    last_few_prices = [[] for _ in range(num_airlines)]
    
    # Reset UI elements
    day_label.config(text="Day: 0")
    demand_label.config(text="Total Market Demand: 1000")
    avg_price_var.set("Avg. Market Price: $100.00")
    
    # Clear plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    
    # Redraw empty plots
    fig.tight_layout()
    canvas.draw()
    
    print("Simulation reset to initial state")

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

    # Sort airlines by cumulative profit
    sorted_airlines = sorted(airlines, key=lambda a: final_cumulative_profits[a], reverse=True)
    
    # Construct the message
    message = "Equilibrium Reached!\n\n"
    message += f"The simulation has reached an equilibrium state on Day {current_day}.\n"
    message += f"Reason: Prices have stabilized with minimal changes over the last {price_history_window} days.\n\n"
    message += "Final State:\n"
    
    for rank, airline in enumerate(sorted_airlines, 1):
        message += f"Rank {rank}: {airline}\n"
        message += f"  Price: ${final_prices[airline]:.2f}\n"
        message += f"  Profit: ${final_profits[airline]:.2f}\n"
        message += f"  Demand: {final_demands[airline]:.2f} customers\n"
        message += f"  Cumulative Profit: ${final_cumulative_profits[airline]:.2f}\n\n"

    # Display pop-up
    messagebox.showinfo("Equilibrium Reached", message)

    # Save final state to CSV
    final_data = {
        "Airline": sorted_airlines,
        "Rank": list(range(1, len(airlines) + 1)),
        "Final_Price": [final_prices[a] for a in sorted_airlines],
        "Final_Profit": [final_profits[a] for a in sorted_airlines],
        "Final_Demand": [final_demands[a] for a in sorted_airlines],
        "Cumulative_Profit": [final_cumulative_profits[a] for a in sorted_airlines]
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

    # Calculate average market price
    avg_market_price = np.mean(prices)
    avg_price_var.set(f"Avg. Market Price: ${avg_market_price:.2f}")
    
    # Simple price sensitivity model - exponential utility
    exp_utils = np.exp(-alpha * prices)
    demand_shares = exp_utils / np.sum(exp_utils)
    
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

    # Plot 1: Price Trajectories with price range zones
    ax1.axhspan(65, 95, color='#DDFFDD', alpha=0.3, label='Favorable price range')
    ax1.axhspan(95, 120, color='#FFFFDD', alpha=0.3, label='Medium price range')
    ax1.axhspan(120, 180, color='#FFDDDD', alpha=0.3, label='Expensive price range')
    
    for i, airline in enumerate(airlines):
        ax1.plot(days, data[airline]['prices'], label=airline, color=colors[i])
    ax1.set_title("Airline Price Trajectories")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, num_days)
    ax1.set_ylim(60, 180)

    # Plot 2: Profit Trajectories
    for i, airline in enumerate(airlines):
        ax2.plot(days, data[airline]['profits'], label=airline, color=colors[i])
    ax2.set_title("Airline Profit Trajectories")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Profit ($)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, num_days)
    if len(days) > 0:
        max_profit = max([max(data[airline]['profits'] or [0]) for airline in airlines]) * 1.1
        min_profit = min([min(data[airline]['profits'] or [0]) for airline in airlines]) * 1.1
        if max_profit > 0:
            ax2.set_ylim(min_profit, max_profit)

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
    if len(days) > 0:
        max_cum_profit = max([max(data[airline]['cumulative_profits'] or [0]) for airline in airlines])
        if max_cum_profit > 0:
            ax4.set_ylim(0, max_cum_profit * 1.1)

    # Adjust layout and redraw
    fig.tight_layout()
    canvas.draw()

    # Update day and demand labels
    day_label.config(text=f"Day: {current_day}")
    demand_label.config(text=f"Total Market Demand: {total_demand:.2f}")

    # Check for equilibrium
    if check_equilibrium() and current_day > 25:
        is_running = False
        show_equilibrium_message()
        return

    # Rank airlines based on profits (not cumulative profits for short-term focus)
    # This makes airlines react to immediate profit conditions
    profit_ranks = np.argsort(np.argsort(-profits))  # Descending order (rank 0 = highest profit)
    
    # Update last place tracking
    for i, airline in enumerate(airlines):
        if profit_ranks[i] == num_airlines - 1:  # If in last place
            last_place_days[airline] += 1
        else:
            last_place_days[airline] = 0  # Reset if not in last place

    # Adjust prices for the next day
    new_prices = np.zeros(num_airlines)
    
    for i, airline in enumerate(airlines):
        profit = profits[i]
        demand = demands[i]
        own_price = prices[i]
        rank = profit_ranks[i]  # Rank based on current profit (0 = best)
        
        # SIMPLIFIED PRICING STRATEGY:
        # 1. If you're the profit leader, gradually increase prices
        # 2. If you're in last place, drastically cut prices
        # 3. Otherwise, adjust based on rank (worse rank = lower prices)
        
        if rank == 0:  # Market leader - increase prices to maximize profit
            # Leaders should be more aggressive about raising prices
            new_prices[i] = own_price * (1.03 + np.random.uniform(0, 0.02))
            print(f"Day {current_day}: {airline} (leader) increasing price from ${own_price:.2f} to ${new_prices[i]:.2f}")
            
        elif rank == num_airlines - 1:  # Last place - cut prices significantly
            if last_place_days[airline] >= last_place_threshold:
                # Drastic price cut to gain market share
                min_competitor_price = min(np.delete(prices, i))
                # Undercut the lowest competitor by 5-10%
                new_prices[i] = min_competitor_price * (0.9 + np.random.uniform(0, 0.05))
                print(f"Day {current_day}: {airline} (last place) drastically cutting price from ${own_price:.2f} to ${new_prices[i]:.2f}")
            else:
                # Still significant price cut
                new_prices[i] = own_price * (0.93 + np.random.uniform(0, 0.02))
                
        else:  # Middle ranks - adjust based on position
            # Linear relationship: worse rank = more aggressive price cuts
            rank_factor = rank / (num_airlines - 2)  # 0 for 2nd place, 1 for 2nd-to-last
            # More aggressive price reductions for lower ranks
            price_adj = 1.01 - (rank_factor * 0.07)  # 1.01 for 2nd place, 0.94 for 2nd-to-last
            new_prices[i] = own_price * (price_adj + np.random.uniform(-0.01, 0.01))

        # Add more randomness to create volatility and break patterns
        new_prices[i] += np.random.normal(0, 2.0)  # Increased randomness

        # Ensure prices don't go below cost (only hard constraint)
        new_prices[i] = max(new_prices[i], base_costs[i] + 0.01)
        # Upper limit just to keep the simulation reasonable
        new_prices[i] = min(new_prices[i], 180)

    prices = new_prices
    current_day += 1

    # Periodically introduce market disruptions to break equilibrium
    if current_day % 15 == 0:  # Market disruption every 15 days
        # Simulate a market shock that affects all airlines
        shock_magnitude = np.random.uniform(0.85, 1.15)  # More extreme shocks
        prices = prices * shock_magnitude
        print(f"Day {current_day}: Market disruption! All airline prices adjusted by factor {shock_magnitude:.2f}")
        
        # Notify user about the market disruption
        if is_running:
            messagebox.showinfo("Market Disruption", 
                               f"Day {current_day}: Market disruption has occurred!\n"
                               f"All airline prices adjusted by {(shock_magnitude-1)*100:.1f}%.")

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

# Reset Button
reset_button = tk.Button(control_frame, text="Reset Simulation", command=reset_simulation)
reset_button.pack(side=tk.LEFT, padx=5)

# Run the GUI
root.mainloop()

# Save static plots at the end
if current_day > 0:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    
    days = list(range(len(data[airlines[0]]['prices'])))
    
    # Plot 1: Price Trajectories with price range zones
    ax1.axhspan(65, 95, color='#DDFFDD', alpha=0.3, label='Favorable price range')
    ax1.axhspan(95, 120, color='#FFFFDD', alpha=0.3, label='Medium price range')
    ax1.axhspan(120, 180, color='#FFDDDD', alpha=0.3, label='Expensive price range')
    
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
    plt.savefig('simulation_results.png')
    print("Static plot saved as 'simulation_results.png'.")