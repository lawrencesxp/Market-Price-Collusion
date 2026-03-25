@echo off
cd /d d:\Market-Price-Collusion\exports

REM Rename folders
ren loveofgod QLearning_CollusionBands
ren qlearning QLearning_StagedLearning
ren qlearning2 QLearning_RandomWalk
ren Agressive RuleBased_AggressiveUndercutting
ren new RuleBased_PriceMatching
ren please RuleBased_StabilitySticking
ren sustainprofit Interactive_ProfitTargeting
ren user Interactive_PriceLeadership
ren loveofgod_priceshock TacitCollusion_CostShock
ren loveofgod_demandshock TacitCollusion_DemandShock

echo Folders renamed.

REM Rename PNG files in each folder
cd QLearning_CollusionBands
ren loveofgod_300dpi.png QLearning_CollusionBands_300dpi.png
cd ..

cd QLearning_StagedLearning
ren qlearning_300dpi.png QLearning_StagedLearning_300dpi.png
cd ..

cd QLearning_RandomWalk
ren qlearning2_300dpi.png QLearning_RandomWalk_300dpi.png
cd ..

cd RuleBased_AggressiveUndercutting
ren agressive_300dpi.png RuleBased_AggressiveUndercutting_300dpi.png
cd ..

cd RuleBased_PriceMatching
ren new_300dpi.png RuleBased_PriceMatching_300dpi.png
cd ..

cd RuleBased_StabilitySticking
ren please_300dpi.png RuleBased_StabilitySticking_300dpi.png
cd ..

cd Interactive_ProfitTargeting
ren sustainprofit_300dpi.png Interactive_ProfitTargeting_300dpi.png
cd ..

cd Interactive_PriceLeadership
ren user_300dpi.png Interactive_PriceLeadership_300dpi.png
cd ..

cd TacitCollusion_CostShock
ren priceshock_300dpi.png TacitCollusion_CostShock_300dpi.png
cd ..

cd TacitCollusion_DemandShock
ren demandshock_300dpi.png TacitCollusion_DemandShock_300dpi.png
cd ..

echo PNG files renamed.
echo Completed successfully!
