#!/usr/bin/env powershell
# Script to rename export folders and files

cd "d:\Market-Price-Collusion\exports"

# Define mapping of old names to new names
$folderMappings = @{
    "loveofgod" = "QLearning_CollusionBands"
    "qlearning" = "QLearning_StagedLearning"
    "qlearning2" = "QLearning_RandomWalk"
    "Agressive" = "RuleBased_AggressiveUndercutting"
    "new" = "RuleBased_PriceMatching"
    "please" = "RuleBased_StabilitySticking"
    "sustainprofit" = "Interactive_ProfitTargeting"
    "user" = "Interactive_PriceLeadership"
    "loveofgod_priceshock" = "TacitCollusion_CostShock"
    "loveofgod_demandshock" = "TacitCollusion_DemandShock"
}

$imageMappings = @{
    "loveofgod_300dpi.png" = "QLearning_CollusionBands_300dpi.png"
    "qlearning_300dpi.png" = "QLearning_StagedLearning_300dpi.png"
    "qlearning2_300dpi.png" = "QLearning_RandomWalk_300dpi.png"
    "agressive_300dpi.png" = "RuleBased_AggressiveUndercutting_300dpi.png"
    "new_300dpi.png" = "RuleBased_PriceMatching_300dpi.png"
    "please_300dpi.png" = "RuleBased_StabilitySticking_300dpi.png"
    "sustainprofit_300dpi.png" = "Interactive_ProfitTargeting_300dpi.png"
    "user_300dpi.png" = "Interactive_PriceLeadership_300dpi.png"
    "priceshock_300dpi.png" = "TacitCollusion_CostShock_300dpi.png"
    "demandshock_300dpi.png" = "TacitCollusion_DemandShock_300dpi.png"
}

# Rename folders
Write-Host "Renaming export folders..."
foreach ($oldName in $folderMappings.Keys) {
    $newName = $folderMappings[$oldName]
    if (Test-Path $oldName) {
        Rename-Item -Path $oldName -NewName $newName
        Write-Host "  [OK] $oldName -> $newName"
    } else {
        Write-Host "  [SKIP] $oldName not found"
    }
}

Write-Host "`nRenaming PNG files..."
Get-ChildItem -Directory | ForEach-Object {
    $folder = $_
    foreach ($oldFile in $imageMappings.Keys) {
        $newFile = $imageMappings[$oldFile]
        $filePath = Join-Path $folder.FullName $oldFile
        if (Test-Path $filePath) {
            $newPath = Join-Path $folder.FullName $newFile
            Rename-Item -Path $filePath -NewName $newFile
            Write-Host "  [OK] $($folder.Name)/$oldFile -> $newFile"
        }
    }
}

Write-Host "`nDone!"
