#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

exports_dir = Path("d:/Market-Price-Collusion/exports")

# Define folder renames
folder_map = {
    "loveofgod": "QLearning_CollusionBands",
    "qlearning": "QLearning_StagedLearning",
    "qlearning2": "QLearning_RandomWalk",
    "Agressive": "RuleBased_AggressiveUndercutting",
    "new": "RuleBased_PriceMatching",
    "please": "RuleBased_StabilitySticking",
    "sustainprofit": "Interactive_ProfitTargeting",
    "user": "Interactive_PriceLeadership",
    "loveofgod_priceshock": "TacitCollusion_CostShock",
    "loveofgod_demandshock": "TacitCollusion_DemandShock"
}

# Define file renames  
file_map = {
    "loveofgod_300dpi.png": "QLearning_CollusionBands_300dpi.png",
    "qlearning_300dpi.png": "QLearning_StagedLearning_300dpi.png",
    "qlearning2_300dpi.png": "QLearning_RandomWalk_300dpi.png",
    "agressive_300dpi.png": "RuleBased_AggressiveUndercutting_300dpi.png",
    "new_300dpi.png": "RuleBased_PriceMatching_300dpi.png",
    "please_300dpi.png": "RuleBased_StabilitySticking_300dpi.png",
    "sustainprofit_300dpi.png": "Interactive_ProfitTargeting_300dpi.png",
    "user_300dpi.png": "Interactive_PriceLeadership_300dpi.png",
    "priceshock_300dpi.png": "TacitCollusion_CostShock_300dpi.png",
    "demandshock_300dpi.png": "TacitCollusion_DemandShock_300dpi.png"
}

print("Starting rename operations...")
print("=" * 60)

# Step 1: Rename folders
print("\n[STEP 1] Renaming folders...")
for old_name, new_name in folder_map.items():
    old_path = exports_dir / old_name
    new_path = exports_dir / new_name
    
    if old_path.exists() and not new_path.exists():
        try:
            os.rename(old_path, new_path)
            print(f"  [OK] {old_name} -> {new_name}")
        except Exception as e:
            print(f"  [ERROR] Failed to rename {old_name}: {e}")
    elif new_path.exists():
        print(f"  [SKIP] {new_name} already exists")
    else:
        print(f"  [SKIP] {old_name} not found")

# Step 2: Rename PNG files
print("\n[STEP 2] Renaming PNG files...")
for folder_path in exports_dir.iterdir():
    if folder_path.is_dir():
        for old_file, new_file in file_map.items():
            old_file_path = folder_path / old_file
            new_file_path = folder_path / new_file
            
            if old_file_path.exists() and not new_file_path.exists():
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"  [OK] {folder_path.name}/{old_file} -> {new_file}")
                except Exception as e:
                    print(f"  [ERROR] Failed: {e}")

print("\n" + "=" * 60)
print("Rename operations completed!")
