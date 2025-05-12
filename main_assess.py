import math
import random
import pandas as pd
import numpy as np

# Import common modules from your project.
from input_excel import load_parameters_from_excel
from prep_optimization_excel import create_design_space
from optimization import objective_function
from data_manager import get_data_point

# =============================================================================
# 1. Load Shared Parameters from Excel
# =============================================================================
file_path = "Battery_Design_Parameters.xlsx"
(vehicle_scope, battery_scope, carculator_scope, energy_storage, 
 scope, target_function, num_objectives, impact_thresholds) = load_parameters_from_excel(file_path)

# =============================================================================
# 2. Create Design Space (for the entire battery)
# =============================================================================
cells_per_pack, cells_in_series_baseline, cells_in_parallel_baseline, mass_per_cell = create_design_space(
    target_energy=get_data_point("battery_scope", "target_energy"),
    target_voltage=get_data_point("battery_scope", "target_voltage"),
    number_of_packs=get_data_point("battery_scope", "number_of_packs")
)

print("\n--- Base Design Space ---")
print(f"Cells per pack (baseline): {cells_per_pack}")
print(f"Cells in series (baseline): {cells_in_series_baseline}")
print(f"Cells in parallel (baseline): {cells_in_parallel_baseline}")
print(f"Mass per cell: {mass_per_cell} kg\n")

# =============================================================================
# 3. Hard-Code a Test Design (Individual)
# =============================================================================
# For testing, we define our own design variables.
# These values can be adjusted as needed to explore a particular region.

modules_in_series = 6         # Example: 3 modules in series
modules_in_parallel = 1        # Example: 10 modules in parallel
module_design = 3               # 3 => closed non-filled design

module_components = ["cell", "module_voltage_sensor", "module_temperature_sensor", "module_pcb", "module_bms", "module_signal_connector"]
pack_components = ["module", "pack_voltage_sensor", "pack_current_sensor", "pack_pcb", "pack_bms", "pack_signal_connector", "pack_power_connector", "pack_busbar", "pack_relay", "pack_fuse"]

replaceability_module = [1, 0, 0, 0, 0, 0]
replaceability_pack = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# Compute derived values:
modules_per_pack = modules_in_series * modules_in_parallel
print(f"modules per pack: {modules_per_pack}")
cells_per_module = cells_per_pack / modules_per_pack
print(f"Derived cells per module: {cells_per_module}")

# =============================================================================
# 4. Evaluate the Objective Function Directly (No GA Loop)
# =============================================================================
# Call the objective_function with your hard-coded values.
# Note: The objective_function signature is assumed to be:
#    objective_function(scope, target_function, impact_thresholds, 
#         vehicle_scope, carculator_scope, module_design,
#         cells_per_module, cells_per_pack, modules_per_pack, modules_in_parallel,
#         replaceability_module, replaceability_pack, energy_storage)
#
# For testing, we call the raw objective_function directly.
normalized_fit, impact_details, lca_system_model = objective_function(
        scope=scope,
        target_function=target_function,
        impact_thresholds=impact_thresholds,
        vehicle_scope=vehicle_scope,
        carculator_scope=carculator_scope,
        module_design=module_design,
        cells_per_module=cells_per_module,
        cells_per_pack=cells_per_pack,
        modules_per_pack=modules_per_pack,
        modules_in_parallel=modules_in_parallel,
        replaceability_module=replaceability_module,
        replaceability_pack=replaceability_pack,
        replaceable_mounting_add=get_data_point("battery_scope", "replaceable_mounting_add"),
        packing_efficiency=get_data_point("battery_scope", "packing_efficiency"),
        number_of_packs=get_data_point("battery_scope", "number_of_packs"),
        energy_storage=energy_storage
    )

# =============================================================================
# 5. Print (or Save) the Results
# =============================================================================
print("\n--- Objective Function Evaluation ---")
if num_objectives == 1:
    # For single-objective, result expected as ((normalized_od_impact,), od_details, lca_system_model)
    print(f"Normalized Impact: {normalized_fit[0]}")
    print(f"Impact Details: {impact_details}")
    print("LCA System Model Parameters:")
    for key, value in lca_system_model.items():
        print(f"  {key}: {value}")
else:
    # For multi-objective, result expected as ((normalized_cc_impact, normalized_od_impact), impact_details, lca_system_model)
    print(f"Normalized Impacts: {normalized_fit}")
    print(f"Impact Details: {impact_details}")
    print("LCA System Model Parameters:")
    for key, value in lca_system_model.items():
        print(f"  {key}: {value}")

# Optionally, write the results to a CSV/Excel for further assessment.
# For example, you can do:
df_result = pd.DataFrame({
    "Modules in Series": [modules_in_series],
    "Modules in Parallel": [modules_in_parallel],
    "Module Design": [module_design],
    "Replaceability Module": [replaceability_module],
    "Replaceability Pack": [replaceability_pack],
    "Cells per Module": [cells_per_module],
    "Normalized Impact": [normalized_fit if isinstance(normalized_fit, str) else normalized_fit[0]],
    "Impact Details": [str(impact_details)]
})
df_result.to_csv("hardcoded_design_results.csv", index=False)
print("\nResults saved to hardcoded_design_results.csv")