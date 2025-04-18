import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from matplotlib.figure import Figure

# Set random seed for reproducibility
np.random.seed(17)

# Configuration
num_days = 100
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
# Base costs (minimum amount to make profit is $65)
base_costs = np.ones(num_airlines) * 65
# Much higher price sensitivity across all price ranges
# This makes customers strongly prefer cheaper airlines
alpha = 1.5  # Single high value for strong price sensitivity
market_demand_per_day = 1000
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']  # Colors for each airline

# Seasonality setup - more pronounced seasonality (±15% effect)
seasonality_effect = np.sin(np.linspace(0, 3 * np.pi, num_days)) * 0.15

# Initialize data storage
data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': [], 
                 'market_share': [], 'profit_change': []} for airline in airlines}
current_day = 0
is_running = False
price_history_window = 10  # Look at last 10 days for equilibrium check
price_change_threshold = 1.0  # Price changes less than $1 considered stable
last_place_days = {airline: 0 for airline in airlines}  # Track days in last place
last_place_threshold = 10  # More aggressive - take action after just 10 days in last place

# Initialize prices for the first day (as requested: 100, 110, 120, 130, 140)
prices = np.array([100, 110, 120, 130, 140])
last_few_prices = [[] for _ in range(num_airlines)]  # Store recent prices for equilibrium check

# Performance tracking for adaptive strategies
market_position = {airline: 0 for airline in airlines}  # Current rank in the market
profit_trend = {airline: [] for airline in airlines}  # Track profit trends for each airline
demand_trend = {airline: [] for airline in airlines}  # Track demand trends for each airline
price_change_direction = {airline: 0 for airline in airlines}  # Track if prices are trending up or down

# Flag for user-controlled airline (Airline_5)
user_controlled_airline = "Airline_5"
user_control_enabled = True
pending_user_price = None  # Store the user's pending price decision

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

# User control frame for Airline_5
user_control_frame = tk.Frame(root)
user_control_frame.pack(side=tk.BOTTOM, fill=tk.X)

user_price_var = tk.StringVar(value=f"Airline_5 Price: ${prices[4]:.2f}")
user_price_label = tk.Label(user_control_frame, textvariable=user_price_var)
user_price_label.pack(side=tk.LEFT, padx=5)

user_profit_var = tk.StringVar(value="Profit: $0.00")
user_profit_label = tk.Label(user_control_frame, textvariable=user_profit_var)
user_profit_label.pack(side=tk.LEFT, padx=5)

user_demand_var = tk.StringVar(value="Demand: 0 customers")
user_demand_label = tk.Label(user_control_frame, textvariable=user_demand_var)
user_demand_label.pack(side=tk.LEFT, padx=5)

user_rank_var = tk.StringVar(value="Rank: -")
user_rank_label = tk.Label(user_control_frame, textvariable=user_rank_var)
user_rank_label.pack(side=tk.LEFT, padx=5)

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

# Function to calculate customer preference factor based on price
# Higher prices get penalized more severely with exponential penalty
def get_customer_preference(price, avg_market_price):
    # Base price sensitivity factor
    price_factor = np.exp(-alpha * price)
    
    # Additional penalty for prices above market average
    if price > avg_market_price:
        # Exponential penalty increases as price exceeds market average
        relative_markup = (price - avg_market_price) / avg_market_price
        price_factor *= np.exp(-relative_markup * 2)
    
    return price_factor

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
    message += f"Reason: Prices have stabilized with minimal changes over the last {price_history_window} days.\n\n"
    message += "Final State:\n"
    
    # Sort airlines by cumulative profit
    sorted_airlines = sorted(airlines, key=lambda a: final_cumulative_profits[a], reverse=True)
    
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

# Function to prompt user for price change
def get_user_price_input():
    global pending_user_price
    
    if not user_control_enabled or not is_running:
        return
        
    current_price = prices[4]  # Airline_5's current price
    
    # Calculate some helpful metrics to show the user
    avg_price = np.mean(prices)
    min_price = min(prices)
    max_price = max(prices)
    
    # Show a dialog with current market information
    price_info = f"Current Market:\n"
    price_info += f"Your price: ${current_price:.2f}\n"
    price_info += f"Market average: ${avg_price:.2f}\n"
    price_info += f"Lowest competitor: ${min_price:.2f}\n"
    price_info += f"Highest competitor: ${max_price:.2f}\n\n"
    price_info += f"Enter new price (min: $65.00):"
    
    # Get user input for new price
    new_price = simpledialog.askfloat("Set Price for Airline_5", 
                                      price_info,
                                      minvalue=65.0, 
                                      maxvalue=200.0,
                                      initialvalue=current_price)
    
    if new_price is not None:
        pending_user_price = new_price
        user_price_var.set(f"Airline_5 Price: ${pending_user_price:.2f}")

# Function to simulate one day and update the plots
def simulate_day():
    global current_day, prices, is_running, pending_user_price

    if not is_running or current_day >= num_days:
        return

    # Apply user price change if pending
    if pending_user_price is not None and user_controlled_airline == "Airline_5":
        prices[4] = pending_user_price
        pending_user_price = None

    # Seasonality multiplier for the day
    seasonality_multiplier = 1 + seasonality_effect[current_day]
    total_demand = market_demand_per_day * seasonality_multiplier

    # Calculate average market price (excluding any extremely high outliers)
    sorted_prices = np.sort(prices)
    avg_market_price = np.mean(sorted_prices[:4])  # Use the 4 lowest prices to avoid outlier influence
    avg_price_var.set(f"Avg. Market Price: ${avg_market_price:.2f}")
    
    # Calculate customer preference factors based on prices
    preference_factors = np.zeros(num_airlines)
    for i in range(num_airlines):
        preference_factors[i] = get_customer_preference(prices[i], avg_market_price)
    
    # Calculate demand shares using the enhanced preference model
    demand_shares = preference_factors / np.sum(preference_factors)
    
    # Compute daily demand
    demands = demand_shares * total_demand

    # Compute profits and store data
    profits = np.zeros(num_airlines)
    for i, airline in enumerate(airlines):
        price = prices[i]
        demand = demands[i]
        cost = base_costs[i]
        profit = (price - cost) * demand
        market_share = demand / total_demand * 100  # As percentage

        # Store data
        data[airline]['prices'].append(price)
        data[airline]['profits'].append(profit)
        data[airline]['demands'].append(demand)
        data[airline]['market_share'].append(market_share)
        
        # Calculate profit change percentage
        if len(data[airline]['profits']) > 1 and data[airline]['profits'][-2] != 0:
            prev_profit = data[airline]['profits'][-2]
            profit_change_pct = (profit - prev_profit) / abs(prev_profit) * 100 if prev_profit != 0 else 0
            data[airline]['profit_change'].append(profit_change_pct)
        else:
            data[airline]['profit_change'].append(0)
        
        # Update cumulative profits
        if len(data[airline]['cumulative_profits']) == 0:
            data[airline]['cumulative_profits'].append(profit)
        else:
            data[airline]['cumulative_profits'].append(data[airline]['cumulative_profits'][-1] + profit)
        
        profits[i] = profit

        # Store recent prices for equilibrium check
        last_few_prices[i].append(price)
        
        # Track price change direction
        if len(data[airline]['prices']) > 1:
            price_diff = data[airline]['prices'][-1] - data[airline]['prices'][-2]
            price_change_direction[airline] = 1 if price_diff > 0 else (-1 if price_diff < 0 else 0)
            
        # Update profit and demand trends (for more intelligent airline decisions)
        if len(data[airline]['profits']) >= 3:
            recent_profits = data[airline]['profits'][-3:]
            profit_trend[airline] = np.polyfit(range(3), recent_profits, 1)[0]  # Slope of trend
        else:
            profit_trend[airline] = 0
            
        if len(data[airline]['demands']) >= 3:
            recent_demands = data[airline]['demands'][-3:]
            demand_trend[airline] = np.polyfit(range(3), recent_demands, 1)[0]  # Slope of trend
        else:
            demand_trend[airline] = 0
            
    # Update user control display if user is controlling Airline_5
    if user_controlled_airline == "Airline_5":
        user_price_var.set(f"Airline_5 Price: ${prices[4]:.2f}")
        user_profit_var.set(f"Profit: ${profits[4]:.2f}")
        user_demand_var.set(f"Demand: {demands[4]:.0f} customers")
        
        # Calculate rank
        profit_ranks = np.argsort(np.argsort(-profits))  # Descending order (rank 0 = highest profit)
        user_rank_var.set(f"Rank: {profit_ranks[4] + 1} of 5")

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
    max_profit = max([max(data[airline]['profits'] or [0]) for airline in airlines]) * 1.1
    min_profit = min([min(data[airline]['profits'] or [0]) for airline in airlines]) * 1.1
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
    # Set y-limit dynamically based on data
    max_cum_profit = max([max(data[airline]['cumulative_profits'] or [0]) for airline in airlines])
    ax4.set_ylim(0, max_cum_profit * 1.1)

    # Adjust layout and redraw
    fig.tight_layout()
    canvas.draw()

    # Update day and demand labels
    day_label.config(text=f"Day: {current_day}")
    demand_label.config(text=f"Total Market Demand: {total_demand:.2f}")

    # Check for equilibrium
    if check_equilibrium() and current_day > 25:  # Don't check for equilibrium too early
        is_running = False
        show_equilibrium_message()
        return

    # Rank airlines based on cumulative profits
    if current_day > 0:  # Need at least one day of data
        cum_profits = [data[airline]['cumulative_profits'][-1] for airline in airlines]
        # Create rankings (rank 0 = highest profit, rank 4 = lowest profit)
        ranks = np.argsort(np.argsort(cum_profits)[::-1])
        
        # Update airline market positions
        for i, airline in enumerate(airlines):
            market_position[airline] = ranks[i]
            
            # Update last place counter
            if ranks[i] == num_airlines - 1:  # If in last place
                last_place_days[airline] += 1
            else:
                last_place_days[airline] = 0  # Reset if not in last place

    # Adjust prices for the next day - but only for AI-controlled airlines
    new_prices = np.zeros(num_airlines)
    avg_price = np.mean(prices)
    
    # Calculate competitive metrics
    price_min = np.min(prices)
    price_max = np.max(prices)
    price_range = price_max - price_min
    price_median = np.median(prices)
    
    for i, airline in enumerate(airlines):
        # Skip price adjustment for user-controlled airline
        if airline == user_controlled_airline and user_control_enabled:
            new_prices[i] = prices[i]
            continue
            
        profit = profits[i]
        demand = demands[i]
        own_price = prices[i]
        rival_prices = np.delete(prices, i)
        avg_rival_price = np.mean(rival_prices)
        position = market_position.get(airline, 0)  # Market position (0 = first, 4 = last)
        
        # Get profit and demand trends
        p_trend = profit_trend.get(airline, 0)
        d_trend = demand_trend.get(airline, 0)

        # Compute momentum: average price change over the last few days
        momentum = 0
        if len(last_few_prices[i]) >= 3:
            recent_prices = last_few_prices[i][-3:]
            price_changes = np.diff(recent_prices)
            momentum = np.mean(price_changes) * 0.5  # Reduced momentum effect a bit

        # More reactive pricing strategy based on market position and performance trends
        if position == 0:  # Market leader
            if d_trend < -5:  # Losing customers rapidly
                # Cut price to retain customers
                new_prices[i] = own_price * (0.97 + np.random.uniform(0, 0.01))
                print(f"Day {current_day}: {airline} (leader) cutting price to retain customers")
            elif p_trend > 0:  # Profits still growing
                # Continue to increase price slowly
                new_prices[i] = own_price * (1.01 + np.random.uniform(0, 0.01))
            else:  # Profits stagnant or declining
                # Small price reduction to find optimal point
                new_prices[i] = own_price * (0.99 + np.random.uniform(-0.01, 0.01))
                
        elif position == num_airlines - 1:  # Last place
            # More desperate measures when in last place
            if last_place_days[airline] >= last_place_threshold:
                # Drastic action - attempt to undercut the market
                target_price = price_min * 0.95
                # Ensure target doesn't go below cost + small margin
                target_price = max(target_price, base_costs[i] + 5)
                new_prices[i] = target_price
                print(f"Day {current_day}: {airline} taking drastic action! Price from ${own_price:.2f} to ${new_prices[i]:.2f}")
            elif own_price > price_median:
                # If price is above median, reduce aggressively
                reduction_factor = 0.93 + np.random.uniform(0, 0.02)
                new_prices[i] = own_price * reduction_factor
            else:
                # Already below median, try to be slightly cheaper than competitors
                new_prices[i] = price_median * (0.9 + np.random.uniform(0, 0.05))
                    
        else:  # Middle positions 
            relative_position = position / (num_airlines - 1)  # 0 to 1 scale
            
            # Adjust based on profit trend
            if p_trend < -100:  # Significant profit decline
                # Cut price more aggressively
                new_prices[i] = own_price * (0.95 + np.random.uniform(0, 0.02))
                print(f"Day {current_day}: {airline} cutting price due to profit decline")
            elif p_trend > 100:  # Good profit growth
                # Slight price increase
                new_prices[i] = own_price * (1.02 + np.random.uniform(0, 0.01))
            else:
                # Strategic positioning based on market
                if own_price > avg_rival_price * 1.1:
                    # Too expensive compared to rivals
                    new_prices[i] = avg_rival_price * (0.98 + np.random.uniform(0, 0.04))
                elif own_price < avg_rival_price * 0.9:
                    # Much cheaper than rivals, can try to increase
                    if demand > market_demand_per_day * 0.3:  # If demand is strong
                        new_prices[i] = own_price * (1.03 + np.random.uniform(0, 0.02))
                    else:
                        # Keep price advantage
                        new_prices[i] = own_price * (1.01 + np.random.uniform(-0.01, 0.01))
                else:
                    # Similar price to rivals, adjust based on position
                    adjustment = (0.5 - relative_position) * 0.04  # Positive for better positions
                    new_prices[i] = own_price * (1 + adjustment + np.random.uniform(-0.02, 0.02))

        # Add momentum with reduced effect
        new_prices[i] += momentum * 0.5
        
        # Add randomness based on position (more desperate positions = more risk taking)
        # Higher randomness for middle-ranked airlines to break patterns
        if position > 0 and position < num_airlines - 1:
            # Middle positions take more risks
            randomness_factor = 1.0
        else:
            # Leaders and last place are more strategic
            randomness_factor = 0.5
            
        new_prices[i] += np.random.normal(0, randomness_factor)

        # Ensure prices stay within a reasonable range and above cost
        new_prices[i] = np.clip(new_prices[i], base_costs[i] + 0.01, 180)

    prices = new_prices
    current_day += 1

    # Periodically introduce market disruptions to break equilibrium
    if current_day % 25 == 0:
        # Simulate a market shock that affects all airlines
        shock_magnitude = np.random.uniform(0.9, 1.1)
        prices = prices * shock_magnitude
        print(f"Day {current_day}: Market disruption! All prices adjusted by factor {shock_magnitude:.2f}")

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

# Reset functionality
def reset_simulation():
    global current_day, prices, is_running, data, last_place_days, market_position, price_change_direction, last_few_prices
    
    # Stop the simulation if running
    is_running = False
    start_button.config(text="Start Simulation")
    
    # Reset all variables
    current_day = 0
    prices = np.array([100, 110, 120, 130, 140])  # Reset to initial prices
    data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': []} for airline in airlines}
    last_place_days = {airline: 0 for airline in airlines}
    market_position = {airline: 0 for airline in airlines}
    price_change_direction = {airline: 0 for airline in airlines}
    last_few_prices = [[] for _ in range(num_airlines)]
    
    # Clear plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    
    # Reset UI elements
    day_label.config(text="Day: 0")
    demand_label.config(text="Total Market Demand: 1000")
    
    # Redraw empty plots
    fig.tight_layout()
    canvas.draw()
    
    print("Simulation reset to initial state")

# Start Button
start_button = tk.Button(control_frame, text="Start Simulation", command=start_simulation)
start_button.pack(side=tk.LEFT, padx=5)

# Reset Button
reset_button = tk.Button(control_frame, text="Reset Simulation", command=reset_simulation)
reset_button.pack(side=tk.LEFT, padx=5)

# Run the GUI
root.mainloop()

# Save static plots at the end
if current_day > 0:  # Only save if simulation ran
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
    plt.savefig('sustain.png')
    print("Static plot saved as 'enhanced_simulation.png'.")