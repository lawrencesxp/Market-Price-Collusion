"""
Master Export Script - Runs all simulation strategies and exports high-res visualizations
Creates separate folders for each strategy's output
"""

import subprocess
import os
from pathlib import Path

# Define all strategies to export
STRATEGIES = {
    "loveofgod": "Q-Learning with Collusion Rewards (300 days)",
    "qlearning": "Multi-Agent Q-Learning with Progressive Difficulty",
    "qlearning2": "Q-Learning Pre-training + Multi-Agent",
    "Agressive": "Aggressive Rule-Based Strategy",
    "new": "Rule-Based with Equilibrium Detection",
    "please": "Rule-Based with Stability Check",
    "sustainprofit": "User-Controlled vs AI Strategy",
    "user": "User Control Variant",
}

print("="*70)
print("LSXP MARKET PRICE COLLUSION - MULTI-STRATEGY EXPORT")
print("="*70)
print(f"\nExporting {len(STRATEGIES)} strategies...\n")

# Create main exports folder
os.makedirs("exports", exist_ok=True)

for strategy_name, description in STRATEGIES.items():
    print(f"[{'*' * 50}]")
    print(f"Strategy: {strategy_name.upper()}")
    print(f"Description: {description}")
    print("-" * 70)
    
    # Create strategy-specific subfolder
    output_dir = f"exports/{strategy_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create export script for this strategy
    export_script = f"_export_{strategy_name}.py"
    
    print(f"Creating export script: {export_script}")
    print(f"Output directory: {output_dir}/")
    print(f"Status: RUNNING...\n")
    
    # Run the export directly
    try:
        result = subprocess.run(
            ["python", export_script],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"✓ SUCCESS - Files saved to {output_dir}/")
        else:
            print(f"✗ FAILED - Error running {strategy_name}")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT - {strategy_name} took too long")
    except FileNotFoundError:
        print(f"⊘ SKIPPED - {export_script} not found (will create it)\n")

print("\n" + "="*70)
print("All exports completed!")
print("="*70)
print("\nFiles organized by strategy:")
for strategy_name in STRATEGIES.keys():
    output_dir = f"exports/{strategy_name}"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"  {strategy_name}/ - {len(files)} files")
print("\n")
