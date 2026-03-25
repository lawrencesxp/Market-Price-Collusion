import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, FFMpegWriter
from collections import defaultdict
import os

# Set random seed for reproducibility
np.random.seed(17)

# Setup
num_days = 300
num_airlines = 5
airlines = [f"Airline_{i+1}" for i in range(num_airlines)]
base_costs = np.ones(num_airlines) * 75 
alpha = 1.5  # Price sensitivity
market_demand_per_day = 1000

# Professional color palette - SHARP & VIBRANT for presentations
colors = ['#E63946', '#F1FAEE', '#A8DADC', '#457B9D', '#1D3557']  # Better contrast
# Alternative: ['#FF006E', '#FFBE0B', '#8338EC', '#3A86FF', '#FB5607']  # Very vibrant

min_market_share = 0.07

# Q-Learning parameters
price_actions = [-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10]
learning_rate = 0.1
discount_factor = 0.9
exploration_rate_initial = 1.0
exploration_decay = 0.99
exploration_min = 0.05

# Collusion bonus settings
collusion_band_width = 0.075
collusion_reward_bonus = 8000
price_premium_threshold = 100

def get_state(airline_idx, prices, profits, ranks):
    own_price = prices[airline_idx]
    price_tier = 0 if own_price < 85 else 1 if own_price < 110 else 2
    profit = profits[airline_idx]
    profit_tier = 0 if profit < 0 else 1 if profit < 10000 else 2
    rank = ranks[airline_idx]
    avg_price = np.mean(prices)
    relative_price = 0 if own_price < avg_price * 0.9 else 2 if own_price > avg_price * 1.1 else 1
    return (price_tier, profit_tier, rank, relative_price)

def is_in_collusion_band(price, avg_price):
    lower = avg_price * (1 - collusion_band_width)
    upper = avg_price * (1 + collusion_band_width)
    return lower <= price <= upper

def calculate_collusion_bonus(price, all_prices, profits):
    avg_price = np.mean(all_prices)
    if is_in_collusion_band(price, avg_price):
        others_in_band = sum(1 for p in all_prices if is_in_collusion_band(p, avg_price)) - 1
        band_factor = others_in_band / (len(all_prices) - 1) if len(all_prices) > 1 else 0
        if band_factor > 0.8:
            band_factor *= 1.5
        profit_factor = min(1.0, max(0.0, np.mean(profits) / 10000))
        price_premium = 1.0
        if avg_price > price_premium_threshold:
            premium_ratio = (avg_price - price_premium_threshold) / 50
            price_premium = 1.0 + min(2.0, premium_ratio)
        return collusion_reward_bonus * band_factor * profit_factor * price_premium
    return 0

def adjust_demand_shares(demand_shares, min_share=min_market_share):
    total_min_share = min_share * len(demand_shares)
    if total_min_share > 0.95:
        min_share = 0.95 / len(demand_shares)
        total_min_share = 0.95
    remaining_share = 1.0 - total_min_share
    if sum(demand_shares) > 0:
        normalized_shares = demand_shares / sum(demand_shares)
    else:
        normalized_shares = np.ones_like(demand_shares) / len(demand_shares)
    adjusted_shares = min_share + normalized_shares * remaining_share
    adjusted_shares = adjusted_shares / sum(adjusted_shares)
    return adjusted_shares

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
        if np.random.random() < self.exploration_rate:
            return np.random.randint(len(price_actions))
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
    
    def get_action_for_price(self, current_price, action_idx):
        price_change = price_actions[action_idx]
        new_price = current_price * (1 + price_change)
        new_price = max(new_price, base_costs[0] + 0.01)
        new_price = min(new_price, 180)
        return new_price

class AirlineSimulationExport:
    def __init__(self):
        self.q_learning_agents = {airline: QLearningAirline(i) for i, airline in enumerate(airlines)}
        self.current_day = 0
        self.data = {airline: {'prices': [], 'profits': [], 'demands': [], 'cumulative_profits': [], 
                              'market_shares': []} for airline in airlines}
        self.prices = np.array([100, 110, 120, 130, 140])
        self.init_dummy_data()
    
    def init_dummy_data(self):
        exp_utils = np.exp(-alpha * self.prices)
        raw_demand_shares = exp_utils / np.sum(exp_utils)
        demand_shares = adjust_demand_shares(raw_demand_shares)
        demands = demand_shares * market_demand_per_day

        for i, airline in enumerate(airlines):
            price = self.prices[i]
            demand = demands[i]
            cost = base_costs[i]
            profit = (price - cost) * demand
            market_share = demand_shares[i] * 100

            self.data[airline]['prices'].append(price)
            self.data[airline]['profits'].append(profit)
            self.data[airline]['demands'].append(demand)
            self.data[airline]['cumulative_profits'].append(profit)
            self.data[airline]['market_shares'].append(market_share)
    
    def run_full_simulation(self):
        """Run complete simulation without GUI"""
        print("Starting simulation...")
        for day in range(num_days):
            if day % 50 == 0:
                print(f"Day {day}/{num_days}")
            
            # Calculate market conditions
            total_demand = market_demand_per_day
            avg_market_price = np.mean(self.prices)
            
            exp_utils = np.exp(-alpha * self.prices)
            raw_demand_shares = exp_utils / np.sum(exp_utils)
            demand_shares = adjust_demand_shares(raw_demand_shares)
            demands = demand_shares * total_demand

            profits = np.zeros(num_airlines)
            for i, airline in enumerate(airlines):
                price = self.prices[i]
                demand = demands[i]
                cost = base_costs[i]
                profit = (price - cost) * demand
                market_share = demand_shares[i] * 100

                self.data[airline]['prices'].append(price)
                self.data[airline]['profits'].append(profit)
                self.data[airline]['demands'].append(demand)
                self.data[airline]['market_shares'].append(market_share)
                
                if len(self.data[airline]['cumulative_profits']) == 0:
                    self.data[airline]['cumulative_profits'].append(profit)
                else:
                    self.data[airline]['cumulative_profits'].append(
                        self.data[airline]['cumulative_profits'][-1] + profit)
                
                profits[i] = profit

            profit_ranks = np.argsort(np.argsort(-profits))
            new_prices = np.zeros(num_airlines)
            best_airline_idx = np.argmax(profits)
            best_price = self.prices[best_airline_idx]
            
            for i, airline in enumerate(airlines):
                agent = self.q_learning_agents[airline]
                state = get_state(i, self.prices, profits, profit_ranks)
                
                if agent.last_state is not None and agent.last_action is not None:
                    reward = profits[i]
                    reward += calculate_collusion_bonus(self.prices[i], self.prices, profits)
                    agent.learn(agent.last_state, agent.last_action, reward, state)
                
                action_idx = agent.select_action(state)
                new_price = agent.get_action_for_price(self.prices[i], action_idx)
                new_prices[i] = new_price
                
                agent.last_state = state
                agent.last_action = action_idx
            
            self.prices = new_prices
            self.current_day += 1
        
        print("Simulation complete!")
        return self.data
    
    def create_high_res_plot(self, output_filename="loveofgod_simulation.png", dpi=300):
        """Create publication-quality plot"""
        print(f"Creating high-resolution plot ({dpi} DPI)...")
        
        # Create figure with professional styling
        fig = Figure(figsize=(16, 12), dpi=dpi)
        fig.patch.set_facecolor('white')
        
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        days = list(range(len(next(iter(self.data.values()))['prices'])))
        
        # Plot 1: Prices with zones
        ax1.axhspan(65, 95, color='#90EE90', alpha=0.2, label='Favorable')
        ax1.axhspan(95, 120, color='#FFD700', alpha=0.2, label='Medium')
        ax1.axhspan(120, 180, color='#FF6B6B', alpha=0.2, label='Expensive')
        
        avg_price = np.mean(self.prices)
        band_lower = avg_price * (1 - collusion_band_width)
        band_upper = avg_price * (1 + collusion_band_width)
        ax1.axhspan(band_lower, band_upper, color='#4A90E2', alpha=0.15, label='Collusion Band')
        
        for i, airline in enumerate(airlines):
            ax1.plot(days, self.data[airline]['prices'], label=airline, color=colors[i], 
                    linewidth=3, marker='o', markersize=2, markevery=20)
        
        ax1.set_title("Airline Price Trajectories", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Day", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Price ($)", fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10, framealpha=0.95)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(0, num_days)
        ax1.set_ylim(60, 180)
        
        # Plot 2: Profits
        for i, airline in enumerate(airlines):
            ax2.plot(days, self.data[airline]['profits'], label=airline, color=colors[i], 
                    linewidth=3, marker='s', markersize=2, markevery=20)
        
        ax2.set_title("Airline Profit Trajectories", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Day", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Profit ($)", fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10, framealpha=0.95)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(0, num_days)
        
        # Plot 3: Market Share (Stacked area)
        ax3.stackplot(days, *[self.data[airline]['market_shares'] for airline in airlines],
                     labels=airlines, colors=colors, alpha=0.8)
        
        ax3.set_title("Airline Market Share (%)", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Day", fontsize=12, fontweight='bold')
        ax3.set_ylabel("Market Share (%)", fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.95)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim(0, num_days)
        ax3.set_ylim(0, 100)
        
        # Plot 4: Cumulative Profits
        for i, airline in enumerate(airlines):
            ax4.plot(days, self.data[airline]['cumulative_profits'], label=airline, color=colors[i],
                    linewidth=3, marker='^', markersize=2, markevery=20)
        
        ax4.set_title("Cumulative Profits Over Time", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Day", fontsize=12, fontweight='bold')
        ax4.set_ylabel("Cumulative Profit ($)", fontsize=12, fontweight='bold')
        ax4.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xlim(0, num_days)
        
        fig.tight_layout()
        fig.savefig(output_filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_filename}")
        return fig
    
    def create_animation_mp4(self, output_filename="loveofgod_simulation.mp4", fps=10):
        """Create animated MP4 of simulation"""
        print(f"Creating MP4 animation ({fps} fps)...")
        
        days_data = list(range(len(next(iter(self.data.values()))['prices'])))
        
        fig = Figure(figsize=(16, 12), dpi=100)
        fig.patch.set_facecolor('white')
        
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        
        lines1 = [ax1.plot([], [], color=colors[i], linewidth=3, label=airlines[i])[0] 
                 for i in range(num_airlines)]
        lines2 = [ax2.plot([], [], color=colors[i], linewidth=3, label=airlines[i])[0] 
                 for i in range(num_airlines)]
        lines4 = [ax4.plot([], [], color=colors[i], linewidth=3, label=airlines[i])[0] 
                 for i in range(num_airlines)]
        
        # Static elements
        ax1.axhspan(65, 95, color='#90EE90', alpha=0.2)
        ax1.axhspan(95, 120, color='#FFD700', alpha=0.2)
        ax1.axhspan(120, 180, color='#FF6B6B', alpha=0.2)
        ax1.set_title("Airline Price Trajectories", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Day", fontsize=10)
        ax1.set_ylabel("Price ($)", fontsize=10)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, num_days)
        ax1.set_ylim(60, 180)
        
        ax2.set_title("Airline Profit Trajectories", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Day", fontsize=10)
        ax2.set_ylabel("Profit ($)", fontsize=10)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, num_days)
        
        ax3.set_title("Airline Market Share (%)", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Day", fontsize=10)
        ax3.set_ylabel("Market Share (%)", fontsize=10)
        ax3.set_xlim(0, num_days)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        
        ax4.set_title("Cumulative Profits Over Time", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Day", fontsize=10)
        ax4.set_ylabel("Cumulative Profit ($)", fontsize=10)
        ax4.legend(loc='upper left', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, num_days)
        
        def animate(frame):
            day_idx = min(frame, len(days_data) - 1)
            
            # Update price lines
            for i in range(num_airlines):
                lines1[i].set_data(days_data[:day_idx+1], 
                                 self.data[airlines[i]]['prices'][:day_idx+1])
            
            # Update profit lines
            for i in range(num_airlines):
                lines2[i].set_data(days_data[:day_idx+1],
                                 self.data[airlines[i]]['profits'][:day_idx+1])
            
            # Update stacked area for market share
            ax3.clear()
            ax3.stackplot(days_data[:day_idx+1], 
                         *[self.data[airline]['market_shares'][:day_idx+1] for airline in airlines],
                         labels=airlines, colors=colors, alpha=0.8)
            ax3.set_title("Airline Market Share (%)", fontsize=14, fontweight='bold')
            ax3.set_xlabel("Day", fontsize=10)
            ax3.set_ylabel("Market Share (%)", fontsize=10)
            ax3.set_xlim(0, num_days)
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper left', fontsize=9, ncol=2)
            
            # Update cumulative profit lines
            for i in range(num_airlines):
                lines4[i].set_data(days_data[:day_idx+1],
                                 self.data[airlines[i]]['cumulative_profits'][:day_idx+1])
            
            max_profit = max([max(self.data[airline]['profits'][:day_idx+1]) 
                            for airline in airlines]) if day_idx > 0 else 10000
            ax2.set_ylim(0, max_profit * 1.2)
            
            max_cum = max([max(self.data[airline]['cumulative_profits'][:day_idx+1]) 
                          for airline in airlines]) if day_idx > 0 else 10000
            ax4.set_ylim(0, max_cum * 1.1)
            
            return lines1 + lines2 + lines4
        
        anim = FuncAnimation(fig, animate, frames=len(days_data), interval=1000//fps, 
                           blit=True, repeat=False)
        
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(output_filename, writer=writer)
        print(f"✓ Saved: {output_filename}")

# Run everything
if __name__ == "__main__":
    os.makedirs("exports", exist_ok=True)
    
    sim = AirlineSimulationExport()
    sim.run_full_simulation()
    sim.create_high_res_plot("exports/loveofgod_simulation_300dpi.png", dpi=300)
    sim.create_high_res_plot("exports/loveofgod_simulation_150dpi.png", dpi=150)
    sim.create_animation_mp4("exports/loveofgod_simulation.mp4", fps=15)
    
    print("\n" + "="*50)
    print("✓ ALL EXPORTS COMPLETE")
    print("="*50)
    print("Files created:")
    print("  - exports/loveofgod_simulation_300dpi.png (for printing)")
    print("  - exports/loveofgod_simulation_150dpi.png (for PowerPoint)")
    print("  - exports/loveofgod_simulation.mp4 (animation)")
    print("="*50)
