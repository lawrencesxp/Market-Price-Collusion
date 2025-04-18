import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from collections import defaultdict
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
num_days = 100
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
base_costs = np.ones(num_airlines) * 65  # $65 minimum to make profit
alpha = 1.5  # Price sensitivity
market_demand_per_day = 1000
colors = ['red', 'blue', 'green', 'orange', 'purple']  # Brighter colors

# Minimum market share (7% guaranteed for each airline)
min_market_share = 0.07

# Q-Learning parameters
price_actions = [-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10]
learning_rate = 0.1
discount_factor = 0.9
exploration_rate_initial = 1.0
exploration_decay = 0.99
exploration_min = 0.05

# Collusion bonus settings
collusion_band_width = 0.10  # % range around average price
collusion_reward_bonus = 3000  # Bonus for staying in collusion band

# Function to discretize state space
def get_state(airline_idx, prices, profits, ranks):
    """Convert the continuous state into a discrete state for Q-learning."""
    own_price = prices[airline_idx]
    
    # Price tier
    if own_price < 85:
        price_tier = 0  # Low
    elif own_price < 110:
        price_tier = 1  # Medium
    else:
        price_tier = 2  # High
        
    # Profit tier
    profit = profits[airline_idx]
    if profit < 0:
        profit_tier = 0  # Losing money
    elif profit < 10000:
        profit_tier = 1  # Modest profit
    else:
        profit_tier = 2  # High profit
        
    # Rank (0-indexed in the code, but represents positions 1-5)
    rank = ranks[airline_idx]
        
    # Relative price position
    avg_price = np.mean(prices)
    if own_price < avg_price * 0.9:
        relative_price = 0  # Below average
    elif own_price > avg_price * 1.1:
        relative_price = 2  # Above average
    else:
        relative_price = 1  # Near average
        
    # Combine into a state tuple
    return (price_tier, profit_tier, rank, relative_price)

# Check if an airline is within the collusion band
def is_in_collusion_band(price, avg_price):
    """Check if a price is within the collusion band around average."""
    lower = avg_price * (1 - collusion_band_width)
    upper = avg_price * (1 + collusion_band_width)
    return lower <= price <= upper

# Calculate collusion bonus for an airline
def calculate_collusion_bonus(price, all_prices, profits):
    """Calculate bonus reward for colluding behavior."""
    avg_price = np.mean(all_prices)
    
    # Check if price is in the collusion band
    if is_in_collusion_band(price, avg_price):
        # Scale bonus by how many other airlines are also in the band
        others_in_band = sum(1 for p in all_prices if is_in_collusion_band(p, avg_price)) - 1
        band_factor = others_in_band / (len(all_prices) - 1) if len(all_prices) > 1 else 0
        
        # Scale bonus by profitability - higher bonus for profitable collusion
        profit_factor = min(1.0, max(0.0, np.mean(profits) / 10000))
        
        return collusion_reward_bonus * band_factor * profit_factor
    
    return 0

# Adjust demand shares to ensure minimum market share
def adjust_demand_shares(demand_shares, min_share=min_market_share):
    """Ensure each airline gets at least the minimum market share."""
    
    # Calculate how much share to guarantee
    total_min_share = min_share * len(demand_shares)
    
    # If we're guaranteeing too much, scale back
    if total_min_share > 0.95:
        min_share = 0.95 / len(demand_shares)
        total_min_share = 0.95
    
    # Calculate remaining share to distribute based on calculated ratios
    remaining_share = 1.0 - total_min_share
    
    # Calculate normalized shares for the remaining portion
    if sum(demand_shares) > 0:
        normalized_shares = demand_shares / sum(demand_shares)
    else:
        normalized_shares = np.ones_like(demand_shares) / len(demand_shares)
    
    # Combine guaranteed minimum with proportional remainder
    adjusted_shares = min_share + normalized_shares * remaining_share
    
    # Ensure they sum to 1.0
    adjusted_shares = adjusted_shares / sum(adjusted_shares)
    
    return adjusted_shares

# Q-Learning Agent class
class QLearningAirline:
    def __init__(self, airline_id):
        self.airline_id = airline_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate_initial
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.q_table = defaultdict(lambda: np.zeros(len(price_actions)))
        self.last_state = None
        self.last_action = None
    
    def select_action(self, state):
        # Explore: select a random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(len(price_actions))
        
        # Exploit: select the best action from Q-table
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        # Q-learning update formula
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Decay exploration rate
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
    
    def get_action_for_price(self, current_price, action_idx):
        """Convert an action index to an actual price"""
        price_change = price_actions[action_idx]
        new_price = current_price * (1 + price_change)
        
        # Ensure prices don't go below cost or above max price
        new_price = max(new_price, base_costs[0] + 0.01)
        new_price = min(new_price, 180)
        
        return new_price

# Main Simulation class
class AirlineSimulation:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Airline Price Competition Simulation with Q-Learning")
        
        # Initialize simulation variables
        self.q_learning_agents = {airline: QLearningAirline(i) for i, airline in enumerate(airlines)}
        self.current_day = 0
        self.is_running = False
        self.data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': [], 
                              'market_shares': []} for airline in airlines}
        self.prices = np.array([100, 110, 120, 130, 140])  # Initial prices
        
        # Create the GUI
        self.setup_gui()
        
        # Initialize plots with some dummy data to avoid empty plots
        self.init_dummy_data()
    
    def init_dummy_data(self):
        """Initialize with one day of dummy data to establish plot lines"""
        # Calculate demand shares for initial prices
        exp_utils = np.exp(-alpha * self.prices)
        raw_demand_shares = exp_utils / np.sum(exp_utils)
        
        # Apply minimum market share
        demand_shares = adjust_demand_shares(raw_demand_shares)
        
        demands = demand_shares * market_demand_per_day

        # Calculate profits
        for i, airline in enumerate(airlines):
            price = self.prices[i]
            demand = demands[i]
            cost = base_costs[i]
            profit = (price - cost) * demand
            market_share = demand_shares[i] * 100  # as percentage

            # Store initial data point
            self.data[airline]['prices'].append(price)
            self.data[airline]['profits'].append(profit)
            self.data[airline]['demands'].append(demand)
            self.data[airline]['cumulative_profits'].append(profit)
            self.data[airline]['market_shares'].append(market_share)
        
        # Update plots with initial data
        self.update_plots()
    
    def setup_gui(self):
        """Create the GUI"""
        # Create a figure with four subplots
        self.fig = Figure(figsize=(15, 10))
        self.ax1 = self.fig.add_subplot(411)  # Prices
        self.ax2 = self.fig.add_subplot(412)  # Profits
        self.ax3 = self.fig.add_subplot(413)  # Demand shares
        self.ax4 = self.fig.add_subplot(414)  # Cumulative profits

        # Embed the figure in the tkinter window
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Control frame for buttons and sliders
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Speed control slider
        speed_label = tk.Label(control_frame, text="Animation Speed (ms):")
        speed_label.pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.IntVar(value=100)
        speed_slider = tk.Scale(control_frame, from_=10, to=500, orient=tk.HORIZONTAL, variable=self.speed_var)
        speed_slider.pack(side=tk.LEFT, padx=5)

        # Learning toggle
        self.learning_var = tk.BooleanVar(value=True)
        learning_check = tk.Checkbutton(control_frame, text="Continue Learning", variable=self.learning_var)
        learning_check.pack(side=tk.LEFT, padx=10)
        
        # Collusion reward toggle
        self.collusion_var = tk.BooleanVar(value=True)
        collusion_check = tk.Checkbutton(control_frame, text="Enable Collusion Rewards", variable=self.collusion_var)
        collusion_check.pack(side=tk.LEFT, padx=10)

        # Mimicry toggle
        self.mimic_var = tk.BooleanVar(value=True)
        mimic_check = tk.Checkbutton(control_frame, text="Enable Mimicry of Leaders", variable=self.mimic_var)
        mimic_check.pack(side=tk.LEFT, padx=10)

        # Exploration rate display
        self.exploration_label = tk.Label(control_frame, text="Avg Exploration Rate: 1.00")
        self.exploration_label.pack(side=tk.LEFT, padx=10)

        # Current day and market demand display
        info_frame = tk.Frame(self.root)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.day_label = tk.Label(info_frame, text="Day: 0")
        self.day_label.pack(side=tk.LEFT, padx=5)
        self.demand_label = tk.Label(info_frame, text="Total Market Demand: 1000")
        self.demand_label.pack(side=tk.LEFT, padx=5)
        self.avg_price_var = tk.StringVar(value="Avg. Market Price: $100.00")
        avg_price_label = tk.Label(info_frame, textvariable=self.avg_price_var)
        avg_price_label.pack(side=tk.LEFT, padx=5)

        # Buttons
        self.start_button = tk.Button(control_frame, text="Start Simulation", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(control_frame, text="Reset Simulation", command=self.reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=5)
    
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        # Stop the simulation if running
        self.is_running = False
        self.start_button.config(text="Start Simulation")
        
        # Reset simulation variables
        self.current_day = 0
        self.prices = np.array([100, 110, 120, 130, 140])  # Reset to initial prices
        self.data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': [],
                              'market_shares': []} for airline in airlines}
        
        # Reset agents
        self.q_learning_agents = {airline: QLearningAirline(i) for i, airline in enumerate(airlines)}
        
        # Reset UI elements
        self.day_label.config(text="Day: 0")
        self.demand_label.config(text="Total Market Demand: 1000")
        self.avg_price_var.set("Avg. Market Price: $100.00")
        
        # Reinitialize with dummy data
        self.init_dummy_data()
        
        print("Simulation reset to initial state")
    
    def start_simulation(self):
        """Start or pause the simulation"""
        if not self.is_running:
            self.is_running = True
            self.start_button.config(text="Pause Simulation")
            self.simulate_day()
        else:
            self.is_running = False
            self.start_button.config(text="Resume Simulation")
    
    def simulate_day(self):
        """Simulate one day of competition"""
        if not self.is_running or self.current_day >= num_days:
            return

        # Calculate market conditions
        total_demand = market_demand_per_day
        avg_market_price = np.mean(self.prices)
        self.avg_price_var.set(f"Avg. Market Price: ${avg_market_price:.2f}")
        
        # Calculate raw demand shares using price sensitivity
        exp_utils = np.exp(-alpha * self.prices)
        raw_demand_shares = exp_utils / np.sum(exp_utils)
        
        # Apply minimum market share
        demand_shares = adjust_demand_shares(raw_demand_shares)
        
        # Calculate actual demands
        demands = demand_shares * total_demand

        # Calculate profits
        profits = np.zeros(num_airlines)
        for i, airline in enumerate(airlines):
            price = self.prices[i]
            demand = demands[i]
            cost = base_costs[i]
            profit = (price - cost) * demand
            market_share = demand_shares[i] * 100  # as percentage

            # Store data
            self.data[airline]['prices'].append(price)
            self.data[airline]['profits'].append(profit)
            self.data[airline]['demands'].append(demand)
            self.data[airline]['market_shares'].append(market_share)
            
            # Update cumulative profits
            if len(self.data[airline]['cumulative_profits']) == 0:
                self.data[airline]['cumulative_profits'].append(profit)
            else:
                self.data[airline]['cumulative_profits'].append(self.data[airline]['cumulative_profits'][-1] + profit)
            
            profits[i] = profit

        # Rank airlines based on profits
        profit_ranks = np.argsort(np.argsort(-profits))  # 0 = highest profit

        # Update plots
        self.update_plots()

        # Update day and demand labels
        self.current_day += 1  # Increment day counter
        self.day_label.config(text=f"Day: {self.current_day}")
        self.demand_label.config(text=f"Total Market Demand: {total_demand:.2f}")

        # Q-Learning: determine next prices
        new_prices = np.zeros(num_airlines)
        learning_enabled = self.learning_var.get()
        
        # Find best performing airline for mimicry
        best_airline_idx = np.argmax(profits)
        best_price = self.prices[best_airline_idx]
        
        # Use agents to determine next prices
        for i, airline in enumerate(airlines):
            agent = self.q_learning_agents[airline]
            
            # Get current state
            state = get_state(i, self.prices, profits, profit_ranks)
            
            # Learn from previous experience if we have it
            if agent.last_state is not None and agent.last_action is not None:
                reward = profits[i]
                if self.collusion_var.get():
                    reward += calculate_collusion_bonus(self.prices[i], self.prices, profits)
                agent.learn(agent.last_state, agent.last_action, reward, state)
            
            # Determine if this airline should mimic the leader
            should_mimic = False
            if self.mimic_var.get() and profit_ranks[i] >= num_airlines - 2:  # Bottom 2 performers
                # If profit is poor or market share is low, try to mimic the leader
                if profits[i] < profits[best_airline_idx] * 0.5 or demand_shares[i] < 0.1:
                    should_mimic = True
                    # Approach leader's price (not immediately copy)
                    price_diff = best_price - self.prices[i]
                    if abs(price_diff) > 0:
                        # Move 30% of the way towards the leader's price
                        new_price = self.prices[i] + 0.3 * price_diff
                        # Ensure within bounds
                        new_price = max(new_price, base_costs[i] + 0.01)
                        new_price = min(new_price, 180)
                        new_prices[i] = new_price
                        print(f"Day {self.current_day}: {airline} mimicking leader, moving price from ${self.prices[i]:.2f} to ${new_price:.2f}")
            
            # If not mimicking, use Q-learning
            if not should_mimic:
                # Select action for next step
                action_idx = agent.select_action(state)
                
                # Calculate new price based on action
                new_price = agent.get_action_for_price(self.prices[i], action_idx)
                
                # Add exploration noise if learning is enabled
                if learning_enabled and np.random.random() < agent.exploration_rate:
                    new_price += np.random.normal(0, 2.0)
                    new_price = max(new_price, base_costs[i] + 0.01)
                    new_price = min(new_price, 180)
                
                new_prices[i] = new_price
                
                # Store state and action for next iteration
                agent.last_state = state
                agent.last_action = action_idx
        
        # Update exploration rate display
        avg_exploration = np.mean([agent.exploration_rate for agent in self.q_learning_agents.values()])
        self.exploration_label.config(text=f"Avg Exploration Rate: {avg_exploration:.2f}")
        
        # Apply new prices
        self.prices = new_prices
        
        # Schedule next day
        if self.current_day < num_days and self.is_running:
            self.root.after(self.speed_var.get(), self.simulate_day)
        elif self.current_day >= num_days:
            # End of simulation
            self.is_running = False
            self.show_final_results()
    
    def update_plots(self):
        """Update all plots with current data"""
        # Get days array for plotting
        days = list(range(len(next(iter(self.data.values()))['prices'])))
        if not days:
            return
            
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        # Plot 1: Price Trajectories with price range zones
        self.ax1.axhspan(65, 95, color='#DDFFDD', alpha=0.3, label='Favorable')
        self.ax1.axhspan(95, 120, color='#FFFFDD', alpha=0.3, label='Medium')
        self.ax1.axhspan(120, 180, color='#FFDDDD', alpha=0.3, label='Expensive')
        
        # Add collusion band
        avg_price = np.mean(self.prices)
        band_lower = avg_price * (1 - collusion_band_width)
        band_upper = avg_price * (1 + collusion_band_width)
        self.ax1.axhspan(band_lower, band_upper, color='#AADDFF', alpha=0.5, label='Collusion band')
        
        # Plot price lines
        for i, airline in enumerate(airlines):
            self.ax1.plot(days, self.data[airline]['prices'], label=airline, color=colors[i], linewidth=2.5)
        
        self.ax1.set_title("Airline Price Trajectories", fontsize=12, fontweight='bold')
        self.ax1.set_xlabel("Day")
        self.ax1.set_ylabel("Price ($)")
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True)
        self.ax1.set_xlim(0, num_days)
        self.ax1.set_ylim(60, 180)

        # Plot 2: Profit Trajectories
        for i, airline in enumerate(airlines):
            self.ax2.plot(days, self.data[airline]['profits'], label=airline, color=colors[i], linewidth=2.5)
        
        self.ax2.set_title("Airline Profit Trajectories", fontsize=12, fontweight='bold')
        self.ax2.set_xlabel("Day")
        self.ax2.set_ylabel("Profit ($)")
        self.ax2.legend(loc='best')
        self.ax2.grid(True)
        self.ax2.set_xlim(0, num_days)
        
        # Set y-axis limits for profits
        max_profit = max([max(self.data[airline]['profits']) for airline in airlines]) * 1.1
        min_profit = min([min(self.data[airline]['profits']) for airline in airlines]) * 1.1
        self.ax2.set_ylim(min_profit, max_profit)

        # Plot 3: Market Share (%) instead of raw demand
        for i, airline in enumerate(airlines):
            self.ax3.plot(days, self.data[airline]['market_shares'], label=airline, color=colors[i], linewidth=2.5)
        
        self.ax3.set_title("Airline Market Share (%)", fontsize=12, fontweight='bold')
        self.ax3.set_xlabel("Day")
        self.ax3.set_ylabel("Market Share (%)")
        self.ax3.legend(loc='best')
        self.ax3.grid(True)
        self.ax3.set_xlim(0, num_days)
        self.ax3.set_ylim(0, 100)

        # Plot 4: Cumulative Profits
        for i, airline in enumerate(airlines):
            self.ax4.plot(days, self.data[airline]['cumulative_profits'], label=airline, color=colors[i], linewidth=2.5)
        
        self.ax4.set_title("Cumulative Profits Over Time", fontsize=12, fontweight='bold')
        self.ax4.set_xlabel("Day")
        self.ax4.set_ylabel("Cumulative Profit ($)")
        self.ax4.legend(loc='upper left')
        self.ax4.grid(True)
        self.ax4.set_xlim(0, num_days)
        
        # Set y-axis limits for cumulative profits
        max_cum_profit = max([max(self.data[airline]['cumulative_profits']) for airline in airlines]) * 1.1
        self.ax4.set_ylim(0, max_cum_profit)

        # Adjust layout and redraw
        self.fig.tight_layout()
        self.fig.canvas.draw()
    
    def show_final_results(self):
        """Show final results when simulation completes"""
        # Gather final data
        final_prices = {airline: self.data[airline]['prices'][-1] for airline in airlines}
        final_profits = {airline: self.data[airline]['profits'][-1] for airline in airlines}
        final_market_shares = {airline: self.data[airline]['market_shares'][-1] for airline in airlines}
        cumulative_profits = {airline: self.data[airline]['cumulative_profits'][-1] for airline in airlines}
        
        # Calculate average market price at end
        avg_final_price = np.mean([final_prices[a] for a in airlines])
        
        # Calculate price clustering
        price_diffs = [abs(final_prices[a] - avg_final_price) / avg_final_price for a in airlines]
        avg_price_diff = np.mean(price_diffs) * 100  # As percentage
        
        # Sort airlines by cumulative profit
        sorted_airlines = sorted(airlines, key=lambda a: cumulative_profits[a], reverse=True)
        
        # Construct result message
        message = "Simulation Complete!\n\n"
        
        # Add collusion analysis
        message += "Price Collusion Analysis:\n"
        message += f"Average Final Price: ${avg_final_price:.2f}\n"
        message += f"Price Clustering: {avg_price_diff:.1f}% from average\n"
        if avg_price_diff < 5:
            message += "Result: Strong evidence of price collusion\n\n"
        elif avg_price_diff < 10:
            message += "Result: Moderate evidence of price collusion\n\n" 
        else:
            message += "Result: No significant price collusion\n\n"
        
        message += "Final Results:\n\n"
        
        for rank, airline in enumerate(sorted_airlines, 1):
            message += f"Rank {rank}: {airline}\n"
            message += f"  Final Price: ${final_prices[airline]:.2f}\n"
            message += f"  Final Profit: ${final_profits[airline]:.2f}\n"
            message += f"  Final Market Share: {final_market_shares[airline]:.1f}%\n"
            message += f"  Cumulative Profit: ${cumulative_profits[airline]:.2f}\n\n"
        
        # Show results
        messagebox.showinfo("Simulation Complete", message)
        
        # Save final state as CSV
        try:
            os.makedirs("results", exist_ok=True)
            final_data = {
                "Airline": sorted_airlines,
                "Rank": list(range(1, len(airlines) + 1)),
                "Final_Price": [final_prices[a] for a in sorted_airlines],
                "Final_Profit": [final_profits[a] for a in sorted_airlines],
                "Market_Share": [final_market_shares[a] for a in sorted_airlines],
                "Cumulative_Profit": [cumulative_profits[a] for a in sorted_airlines]
            }
            pd.DataFrame(final_data).to_csv("results/final_state.csv", index=False)
            print("Results saved to results/final_state.csv")
        except Exception as e:
            print(f"Failed to save results to CSV: {e}")
    
    def run(self):
        """Run the main application loop"""
        self.root.mainloop()

# Main entry point
if __name__ == "__main__":
    app = AirlineSimulation()
    app.run()