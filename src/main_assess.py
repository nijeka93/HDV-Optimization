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
 scope, target_function, num_objectives, impact_thresholds, weighting_factors) = load_parameters_from_excel(file_path)

# =============================================================================
# 2. Create Design Space (for the entire battery)
# =============================================================================

target_energy=get_data_point("battery_scope", "target_energy")
target_voltage=get_data_point("battery_scope", "target_voltage")
number_of_packs=get_data_point("battery_scope", "number_of_packs")


cells_per_pack, cells_in_series_system, cells_in_parallel_system, mass_per_cell = create_design_space(
    target_energy=target_energy,
    target_voltage=target_voltage,
    number_of_packs=number_of_packs
)

cells_in_parallel_pack = cells_in_parallel_system / number_of_packs

print("\n--- Base Design Space ---")
print(f"Cells per pack (baseline): {cells_per_pack}")
print(f"Cells in series (baseline): {cells_in_series_system}")
print(f"Cells in parallel (baseline): {cells_in_parallel_system}")
print(f"Mass per cell: {mass_per_cell} kg\n")
print(f"cell_to_cell_clearance: {get_data_point("battery_scope", "cell_to_cell_clearance")}")
print(f"replaceable_mounting_add: {get_data_point("battery_scope", "replaceable_mounting_add")}")
print(f"packing_efficiency: {get_data_point("battery_scope", "packing_efficiency")}")

# =============================================================================
# 3. Hard-Code a Test Design (Individual)
# =============================================================================
# For testing, we define our own design variables.
# These values can be adjusted as needed to explore a particular region.

modules_in_series = 1         # Example: 3 modules in series
modules_in_parallel = 6        # Example: 10 modules in parallel
module_design = 3               # 2 => closed non-filled design

module_components = ["cell", "module_voltage_sensor", "module_temperature_sensor", "module_pcb", "module_bms", "module_signal_connector"]
pack_components = ["module", "pack_voltage_sensor", "pack_current_sensor", "pack_pcb", "pack_bms", "pack_signal_connector", "pack_power_connector", "pack_busbar", "pack_relay", "pack_fuse"]

replaceability_module = [1, 0, 0, 0, 0, 0]
replaceability_pack = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

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
        weighting_factors=weighting_factors,
        vehicle_scope=vehicle_scope,
        carculator_scope=carculator_scope,
        module_design=module_design,
        cells_per_module=cells_per_module,
        cells_in_parallel=cells_in_parallel_pack,
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
# Flatten impact_details (list of tuples) into per-category/per-element columns
impact_category_codes = {
    "Climate Change Impact": "CC",
    "Ozone Depletion Impact": "OD",
    "Marine Eutrophication Impact": "MEU",
    "Freshwater Eutrophication Impact": "FEU",
    "Terrestrial Acidification Impact": "TAC",
    "Land Use Impact": "LU",
    "Water Use Impact": "WU",
    "Photochemical Oxidant Formation Impact": "POF",
    "Carcinogenic Human Toxicity Impact": "CHT",
    "Non-Carcinogenic Human Toxicity Impact": "NHT",
    "Freshwater Ecotoxicity Impact": "FET",
    "Ionising Radiation Impact": "IR",
    "Energy Resources Depletion Impact": "ERD",
    "Material Resources Impact": "MR",
}
life_cycle_elements = [
    "glider",
    "powertrain",
    "energy storage",
    "energy chain",
    "maintenance",
    "EOL",
    "road",
    "direct - non-exhaust",
    "direct - exhaust",
]
impact_flat = {}
# impact_details is expected as: [(category_name, total_value, [values_by_element]), ...]
for category_name, total_value, breakdown in impact_details:
    code = impact_category_codes.get(category_name, category_name)
    impact_flat[f"{code}_total"] = [total_value]
    # add per-element breakdown
    for i, element in enumerate(life_cycle_elements):
        value = breakdown[i] if i < len(breakdown) else np.nan
        impact_flat[f"{code}_{element}"] = [value]

df_result = pd.DataFrame({
    "Modules in Series": [modules_in_series],
    "Modules in Parallel": [modules_in_parallel],
    "Module Design": [module_design],
    "Replaceability Module": [replaceability_module],
    "Replaceability Pack": [replaceability_pack],
    "Cells per Module": [cells_per_module],
    "Normalized Impact": [normalized_fit if isinstance(normalized_fit, str) else normalized_fit[0]],
    **impact_flat,
})
df_result.to_csv("hardcoded_design_results.csv", index=False)
print("\nResults saved to hardcoded_design_results.csv")