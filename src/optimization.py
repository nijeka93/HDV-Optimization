from carculator_truck import TruckInputParameters, TruckModel, fill_xarray_from_input_parameters, InventoryTruck
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from prep_optimization_excel import objective_value_retention, objective_efficiency_increase
from data_manager import get_component, get_data_point
from classes import VehicleScope, Battery_Scope, Pack, Cell, StandardComponent

import inspect  # add near other imports

# Debug directory and initializer
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_DIR = MODULE_DIR  # write debug CSVs next to this file for predictability
DEBUG_EFF_ARGS_CSV = os.path.join(DEBUG_DIR, "debug_efficiency_call_args.csv")
DEBUG_RET_ARGS_CSV = os.path.join(DEBUG_DIR, "debug_retention_call_args.csv")
_debug_initialized = False

def _debug_init_once():
    global _debug_initialized
    if _debug_initialized:
        return
    _debug_initialized = True
    try:
        print(f"[debug] __file__: {os.path.abspath(__file__)}")
        print(f"[debug] CWD: {os.getcwd()}")
        print(f"[debug] Will write efficiency args to: {DEBUG_EFF_ARGS_CSV}")
        print(f"[debug] Will write retention  args to: {DEBUG_RET_ARGS_CSV}")
        os.makedirs(os.path.dirname(DEBUG_EFF_ARGS_CSV), exist_ok=True)
        os.makedirs(os.path.dirname(DEBUG_RET_ARGS_CSV), exist_ok=True)
    except Exception as e:
        print(f"[debug] Failed to prepare debug paths: {e}")

def _append_debug_row(csv_path: str, row: dict):
    try:
        df = pd.DataFrame([row])
        # Emit a visible hint on first write
        if not os.path.exists(csv_path):
            print(f"[debug] Creating debug CSV: {os.path.abspath(csv_path)}")
        df.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))
    except Exception as e:
        print(f"[debug][ERROR] Failed writing {csv_path}: {e}")
        print(f"[debug] Offending row: {row}")
        raise

# Combine the elements of the optimization logic: 1) calculate drawbacks 2) pass drawback results to Carculator 3) run LCA 4) return LCA results
def objective_function(
    scope: str,
    target_function: str,
    impact_thresholds: dict,
    weighting_factors: dict,
    vehicle_scope: VehicleScope,
    carculator_scope: dict,
    module_design: int,
    replaceability_module: list[int],
    replaceability_pack: list[int],
    number_of_packs: int,
    cells_per_pack: int,
    cells_per_module: float,
    cells_in_parallel: int,
    modules_in_parallel: int,
    modules_per_pack: int,
    replaceable_mounting_add: float,
    packing_efficiency: float,
    energy_storage: dict):

    _debug_init_once()
    print(f"[debug] objective_function() entered in: {os.path.abspath(__file__)}")
    print("[debug] ... will log prep_optimization_excel inputs")

    print("___________________________________________________")
    print(f"carculator battery dict: {energy_storage}")        
    print("___________________________________________________")    

    # Determine source (optimization vs enumeration) from caller
    try:
        _caller = inspect.stack()[1].filename
    except Exception:
        _caller = "<unknown>"
    _source = "opt" if "main_optimization.py" in str(_caller) else (
        "enum" if "main_full_enumeration.py" in str(_caller) else os.path.basename(str(_caller))
    )

    # --- Debug: log EXACT args going into objective_efficiency_increase ---
    _eff_args = {
        "source": _source,
        "module_design": int(module_design),
        "cell_to_cell_clearance": float(get_data_point("battery_scope", "cell_to_cell_clearance")),
        "sum_rm": int(sum(replaceability_module)),
        "sum_rp": int(sum(replaceability_pack)),
        "cells_per_module": float(cells_per_module),
        "cells_in_parallel": int(cells_in_parallel),
        "cells_per_pack": int(cells_per_pack),
        "modules_per_pack": int(modules_per_pack),
        "modules_in_parallel": int(modules_in_parallel),
        "replaceable_mounting_add": float(replaceable_mounting_add),
        "packing_efficiency": float(packing_efficiency),
        "number_of_packs": int(number_of_packs),
    }
    _append_debug_row(DEBUG_EFF_ARGS_CSV, _eff_args)
    print(f"[debug] WROTE: {DEBUG_EFF_ARGS_CSV} (efficiency inputs)")

    # Call the efficiency objective function
    battery_efficiencies = objective_efficiency_increase(
        module_design=module_design,
        cell_to_cell_clearance=get_data_point("battery_scope", "cell_to_cell_clearance"),
        replaceability_module=replaceability_module,
        replaceability_pack=replaceability_pack,
        cells_per_module=cells_per_module,
        cells_in_parallel=cells_in_parallel,
        cells_per_pack=cells_per_pack,
        modules_per_pack=modules_per_pack,
        modules_in_parallel=modules_in_parallel,
        replaceable_mounting_add=replaceable_mounting_add,
        packing_efficiency=packing_efficiency
    )
    
    battery_cells_mass = battery_efficiencies["cells per pack mass"] * number_of_packs
    battery_total_mass = battery_cells_mass / battery_efficiencies["cell mass share"]
    battery_bop_mass = battery_total_mass * (1 - battery_efficiencies["cell mass share"])
    print(f"optimization l45f: battery cells mass: {battery_cells_mass}")
    print(f"optimization l45f: battery total mass: {battery_total_mass}")
    print(f"optimization l45f: battery BoP mass: {battery_bop_mass}")
    print(f"optimization l45f: replaceable_mounting_add: {replaceable_mounting_add}")
    print(f"optimization l45f: packing_efficiency: {packing_efficiency}")

    # remove stray line that could cause NameError

    # --- Debug: log EXACT args going into objective_value_retention ---
    _ret_args = {
        "source": _source,
        "initial_battery_replacements": float(get_data_point("battery_scope", "initial_battery_replacements")),
        "modules_in_parallel": int(modules_in_parallel),
        "cells_per_module": float(cells_per_module),
        "cells_in_parallel": int(cells_in_parallel),
        "modules_per_pack": int(modules_per_pack),
        "sum_rm": int(sum(replaceability_module)),
        "sum_rp": int(sum(replaceability_pack)),
        "module_mass": float(battery_efficiencies.get("module mass", float("nan"))),
        "pack_mass": float(battery_efficiencies.get("pack mass", float("nan"))),
        "powered_hours": float(get_data_point("vehicle_scope", "powered_hours")),
        "replaceable_mounting_add": float(replaceable_mounting_add),
    }
    _append_debug_row(DEBUG_RET_ARGS_CSV, _ret_args)
    print(f"[debug] WROTE: {DEBUG_RET_ARGS_CSV} (retention inputs)")

    # Call the value retention objective function
    battery_replacements = objective_value_retention(
        initial_battery_replacements=get_data_point("battery_scope", "initial_battery_replacements"),
        modules_in_parallel=modules_in_parallel,
        cells_per_module=cells_per_module,
        cells_in_parallel=cells_in_parallel,
        modules_per_pack=modules_per_pack,
        replaceability_module=replaceability_module,
        replaceability_pack=replaceability_pack,
        module_mass=battery_efficiencies["module mass"],
        pack_mass=battery_efficiencies["pack mass"],
        powered_hours=get_data_point("vehicle_scope", "powered_hours"),
        replaceable_mounting_add=replaceable_mounting_add
        )
       
    # Run LCA
    # 1) Load the default vehicle parameters from carculator
    tip = TruckInputParameters()
    tip.static()
    
    # 2) Fill the xarray with the chosen input parameters
    dcts, array = fill_xarray_from_input_parameters(tip,
        scope=carculator_scope)
    
    # 3) Incorporate translated optimization variables into the LCA model
    array.loc[dict(parameter="lifetime kilometers")] = vehicle_scope.lifetime_kilometers
    array.loc[dict(parameter="battery cell mass share, " + get_data_point("cell", "chemistry"))] = battery_efficiencies["cell mass share"] 
    array.loc[dict(parameter="battery cell energy density, " + get_data_point("cell", "chemistry"))] = [get_data_point("cell", "gravimetric_energy")/1000] # [Wh/kg] --> [kWh/kg]
    array.loc[dict(parameter="battery cell mass")] = battery_cells_mass
    array.loc[dict(parameter="energy battery mass")] = battery_total_mass       
    array.loc[dict(parameter="battery BoP mass")] = battery_bop_mass   
    array.loc[dict(parameter="battery cycle life, " + get_data_point("cell", "chemistry"))] = [get_data_point("cell", "cycle_life")]
    payload_carculator = {
        (vehicle_scope.powertrain, vehicle_scope.size, int(vehicle_scope.year)): int(vehicle_scope.payload)
    }
    power_carculator = {
        (vehicle_scope.powertrain, vehicle_scope.size, int(vehicle_scope.year)): int(vehicle_scope.power)
    }

    # 4) Initialize TruckModel based on user inputs
    tm = TruckModel(
        array,
        cycle = vehicle_scope.driving_cycle,
        target_range = vehicle_scope.target_range,
        country = vehicle_scope.country_of_use,
        payload = payload_carculator,
        power = power_carculator,
        energy_storage = energy_storage
    )
    
    tm.set_all()
    
    print("----------------------------------------------------")    
    print("DEBUG: TruckModel going into calculation:")
    print(tm.array.sel(size='40t', value=0, parameter=['lifetime kilometers', 'battery cell mass share', 'battery cell energy density', 'battery cell mass', 'battery BoP mass', 'battery cycle life']).to_dataframe('val'))
 
    # 5) Adapted values of calculated TruckModel parameters
    tm.array.loc[dict(parameter="battery lifetime replacements", value = 0)] = [battery_replacements] # working!!!; positive impact correlation
    tm.array.loc[dict(parameter="battery cell mass", value = 0)] = [battery_cells_mass] # working!!!; positive impact correlation
    tm.array.loc[dict(parameter="energy battery mass", value = 0)] = [battery_total_mass] # working!!!; positive impact correlation
    tm.array.loc[dict(parameter="battery BoP mass", value = 0)] = [battery_bop_mass] # working!!!; positive impact correlation
    #tm.array.loc[dict(parameter="battery cycle life, " + get_data_point("cell", "chemistry"), value = 0)] = [get_data_point("cell", "cycle_life")]
    
    # 6) Extract additional Carculator parameters
    lca_system_model = {}
    lca_system_model["curb_mass"] = float(tm.array.sel(parameter="curb mass", value=0).data)
    lca_system_model["total_battery_mass"] = float(tm.array.sel(parameter="energy battery mass", value=0).data)
    lca_system_model["battery_cell_mass"] = float(tm.array.sel(parameter="battery cell mass", value=0).data)
    lca_system_model["battery_bop_mass"] = float(tm.array.sel(parameter="battery BoP mass", value=0).data)
    lca_system_model["battery_lifetime_replacements"] = float(tm.array.sel(parameter="battery lifetime replacements", value=0).data)
    lca_system_model["battery_cell_mass_share"] = float(tm.array.sel(parameter="battery cell mass share", value=0).data)
    lca_system_model["battery_cell_mass_share_nmc"] = float(tm.array.sel(parameter="battery cell mass share, " + get_data_point("cell", "chemistry"), value=0).data)
    lca_system_model["TtW_energy"] = float(tm.array.sel(parameter="TtW energy", value=0).data)
    lca_system_model["TtW_efficiency"] = float(tm.array.sel(parameter="TtW efficiency", value=0).data)
    lca_system_model["electricity_consumption"] = float(tm.array.sel(parameter="electricity consumption", value=0).data)
    lca_system_model["battery_cycle_life"] = float(tm.array.sel(parameter="battery cycle life", value=0).data)

    print(f"LCA parameter: 'curb_mass' = ({lca_system_model['curb_mass']})")
    print(f"LCA parameter: 'total_battery_mass' = ({lca_system_model['total_battery_mass']})")
    print(f"LCA parameter: 'battery_cell_mass' = ({lca_system_model['battery_cell_mass']})")    
    print(f"LCA parameter: 'battery_bop_mass' = ({lca_system_model['battery_bop_mass']})")
    print(f"LCA parameter: 'battery_lifetime_replacements' = ({lca_system_model['battery_lifetime_replacements']})")
    print(f"LCA parameter: 'battery_cell_mass_share' = ({lca_system_model['battery_cell_mass_share']})")
    print(f"LCA parameter: 'battery_cell_mass_share_nmc' = ({lca_system_model['battery_cell_mass_share_nmc']})")    
    print(f"LCA parameter: 'TtW_energy' = ({lca_system_model['TtW_energy']})")
    print(f"LCA parameter: 'TtW_efficiency' = ({lca_system_model['TtW_efficiency']})")
    print(f"LCA parameter: 'electricity_consumption' = ({lca_system_model['electricity_consumption']})")
    print(f"LCA parameter: 'battery cycle life' = ({lca_system_model['battery_cycle_life']})")

    # 7) Define impact categories and processes
    impact_categories = ["climate change", "ozone depletion", "eutrophication: marine", "eutrophication: freshwater", 
                         "acidification: terrestrial", "land use", "water use", "photochemical oxidant formation: human health", 
                         "human toxicity: carcinogenic", "human toxicity: non-carcinogenic", "ecotoxicity: freshwater", 
                         "ionising radiation", "energy resources depletion: non-renewable", "material resources: metals/minerals"]
    impact_aspects = ["glider", "powertrain", "energy storage", "energy chain",
                        "maintenance", "EOL", "road", "direct - non-exhaust", "direct - exhaust"]

    # 8) Run LCA calculations to compute environmental impacts and structure results
    ic = InventoryTruck(tm, functional_unit="tkm")
    res = ic.calculate_impacts()
    
    print(f"available impact categories: {ic.impact_categories}")
    
    selected_res = res.sel(impact_category=impact_categories)
    
    reshaped_data = np.squeeze(selected_res.data)
    
    # 9) Process and categorize results
    # a) Verify the reshaped dimensions match expectations
    if reshaped_data.shape != (len(impact_categories), len(impact_aspects)):
        raise ValueError(f"Mismatch: reshaped_data.shape = {reshaped_data.shape}, expected = ({len(impact_categories)}, {len(impact_aspects)})")
    
    impact_results_df = pd.DataFrame(reshaped_data, index=impact_categories, columns=impact_aspects)

    # b) Check for optimization scope, adapting the results according to the set scope:
    if scope == "Battery": # --> considering BoL and electricity (incl. background system) during the use phase
        climate_change_impact = impact_results_df.loc["climate change"]["energy storage"] + impact_results_df.loc["climate change"]["energy chain"]
        ozone_depletion_impact = impact_results_df.loc["ozone depletion"]["energy storage"] + impact_results_df.loc["ozone depletion"]["energy chain"]
        marine_eutrophication_impact = impact_results_df.loc["eutrophication: marine"]["energy storage"] + impact_results_df.loc["eutrophication: marine"]["energy chain"]
        freshwater_eutrophication_impact = impact_results_df.loc["eutrophication: freshwater"]["energy storage"] + impact_results_df.loc["eutrophication: freshwater"]["energy chain"]
        terrestrial_acidification_impact = impact_results_df.loc["acidification: terrestrial"]["energy storage"] + impact_results_df.loc["acidification: terrestrial"]["energy chain"]
        land_use_impact = impact_results_df.loc["land use"]["energy storage"] + impact_results_df.loc["land use"]["energy chain"]
        water_use_impact = impact_results_df.loc["water use"]["energy storage"] + impact_results_df.loc["water use"]["energy chain"]
        photochemical_ozone_formation_impact = impact_results_df.loc["photochemical oxidant formation: human health"]["energy storage"] + impact_results_df.loc["photochemical oxidant formation: human health"]["energy chain"]
        carcinogenic_human_toxicity_impact = impact_results_df.loc["human toxicity: carcinogenic"]["energy storage"] + impact_results_df.loc["human toxicity: carcinogenic"]["energy chain"]
        non_carcinogenic_human_toxicity_impact = impact_results_df.loc["human toxicity: non-carcinogenic"]["energy storage"] + impact_results_df.loc["human toxicity: non-carcinogenic"]["energy chain"]
        freshwater_ecotoxicity_impact = impact_results_df.loc["ecotoxicity: freshwater"]["energy storage"] + impact_results_df.loc["ecotoxicity: freshwater"]["energy chain"]
        ionising_radiation_impact = impact_results_df.loc["ionising radiation"]["energy storage"] + impact_results_df.loc["ionising radiation"]["energy chain"]
        energy_resources_depletion_impact = impact_results_df.loc["energy resources depletion: non-renewable"]["energy storage"] + impact_results_df.loc["energy resources depletion: non-renewable"]["energy chain"]
        material_resources_impact = impact_results_df.loc["material resources: metals/minerals"]["energy storage"] + impact_results_df.loc["material resources: metals/minerals"]["energy chain"]
    else: # --> scope is the entire vehicle: summing the results across all aspects for each impact category
        climate_change_impact = impact_results_df.loc["climate change"].sum()
        ozone_depletion_impact = impact_results_df.loc["ozone depletion"].sum()
        marine_eutrophication_impact = impact_results_df.loc["eutrophication: marine"].sum()
        freshwater_eutrophication_impact = impact_results_df.loc["eutrophication: freshwater"].sum()
        terrestrial_acidification_impact = impact_results_df.loc["acidification: terrestrial"].sum()
        land_use_impact = impact_results_df.loc["land use"].sum()
        water_use_impact = impact_results_df.loc["water use"].sum()
        photochemical_ozone_formation_impact = impact_results_df.loc["photochemical oxidant formation: human health"].sum()
        carcinogenic_human_toxicity_impact = impact_results_df.loc["human toxicity: carcinogenic"].sum()
        non_carcinogenic_human_toxicity_impact = impact_results_df.loc["human toxicity: non-carcinogenic"].sum()
        freshwater_ecotoxicity_impact = impact_results_df.loc["ecotoxicity: freshwater"].sum()
        ionising_radiation_impact = impact_results_df.loc["ionising radiation"].sum()
        energy_resources_depletion_impact = impact_results_df.loc["energy resources depletion: non-renewable"].sum()
        material_resources_impact = impact_results_df.loc["material resources: metals/minerals"].sum()
    
    # Debug print the results
    print(f"Climate Change Impact: {climate_change_impact}")
    print(f"Ozone Depletion Impact: {ozone_depletion_impact}")
    print(f"Marine Eutrophication Impact: {marine_eutrophication_impact}")
    print(f"Freshwater Eutrophication Impact: {freshwater_eutrophication_impact}")
    print(f"Terrestrial Acidification Impact: {terrestrial_acidification_impact}")
    print(f"Land Use Impact: {land_use_impact}")
    print(f"Water Use Impact: {water_use_impact}")
    print(f"Photochemical Ozone Formation Impact: {photochemical_ozone_formation_impact}")
    print(f"Carcinogenic Human Toxicity Impact: {carcinogenic_human_toxicity_impact}")
    print(f"Non-Carcinogenic Human Toxicity Impact: {non_carcinogenic_human_toxicity_impact}")
    print(f"Freshwater Ecotoxicity Impact: {freshwater_ecotoxicity_impact}")
    print(f"Ionising Radiation Impact: {ionising_radiation_impact}")
    print(f"Energy Resources Depletino Impact: {energy_resources_depletion_impact}")
    print(f"Material Resources Impact: {material_resources_impact}")
    
    # c) Prepare penalty, if configuration is infeasible (foam filling but replaceable module components)
    penalty_factor = 1
    if module_design < 2 and any(replaceability_module):
        penalty_count = sum(replaceability_module)
        penalty_factor = 1 + 0.5 * penalty_count

    # Helper mapping for single-objective results
    target_function_mapping = {
        "climate change": ("Climate Change Impact", climate_change_impact, "climate change"),
        "ozone depletion": ("Ozone Depletion Impact", ozone_depletion_impact, "ozone depletion"),
        "eutrophication: marine": ("Marine Eutrophication Impact", marine_eutrophication_impact, "eutrophication: marine"),
        "eutrophication: freshwater": ("Freshwater Eutrophication Impact", freshwater_eutrophication_impact, "eutrophication: freshwater"),
        "acidification: terrestrial": ("Terrestrial Acidification Impact", terrestrial_acidification_impact, "acidification: terrestrial"),
        "land use": ("Land Use Impact", land_use_impact, "land use"),
        "water use": ("Water Use Impact", water_use_impact, "water use"),
        "photochemical oxidant formation: human health": ("Photochemical Ozone Formation Impact", photochemical_ozone_formation_impact, "photochemical oxidant formation: human health"),
        "human toxicity: carcinogenic": ("Carcinogenic Human Toxicity Impact", carcinogenic_human_toxicity_impact, "human toxicity: carcinogenic"),
        "human toxicity: non-carcinogenic": ("Non-Carcinogenic Human Toxicity Impact", non_carcinogenic_human_toxicity_impact, "human toxicity: non-carcinogenic"),
        "ecotoxicity: freshwater": ("Freshwater Ecotoxicity Impact", freshwater_ecotoxicity_impact, "ecotoxicity: freshwater"),
        "ionising radiation": ("Ionising Radiation Impact", ionising_radiation_impact, "ionising radiation"),
        "energy resources depletion: non-renewable": ("Energy Resources Depletion Impact", energy_resources_depletion_impact, "energy resources depletion: non-renewable"),
        "material resources: metals/minerals": ("Material Resources Impact", material_resources_impact, "material resources: metals/minerals"),
    }

    if target_function in target_function_mapping:
        label, impact, category = target_function_mapping[target_function]
        normalized_impact = impact / impact_thresholds[category] * penalty_factor
        print(f"Normalized {label}: {normalized_impact}; (penalized)")
        aspects = impact_results_df.loc[category].tolist()
        return ((normalized_impact,), (category, impact, aspects), lca_system_model)

    elif target_function == "Total Environmental Impact (distance-to-target)":
        normalized_cc_impact = climate_change_impact / impact_thresholds["climate change"] * penalty_factor
        normalized_od_impact = ozone_depletion_impact / impact_thresholds["ozone depletion"] * penalty_factor
        normalized_meu_impact = marine_eutrophication_impact / impact_thresholds["eutrophication: marine"] * penalty_factor
        normalized_feu_impact = freshwater_eutrophication_impact / impact_thresholds["eutrophication: freshwater"] * penalty_factor
        normalized_tac_impact = terrestrial_acidification_impact / impact_thresholds["acidification: terrestrial"] * penalty_factor
        normalized_lu_impact = land_use_impact / impact_thresholds["land use"] * penalty_factor
        normalized_wu_impact = water_use_impact / impact_thresholds["water use"] * penalty_factor
        normalized_pof_impact = photochemical_ozone_formation_impact / impact_thresholds["photochemical oxidant formation: human health"] * penalty_factor
        normalized_cht_impact = carcinogenic_human_toxicity_impact / impact_thresholds["human toxicity: carcinogenic"] * penalty_factor
        normalized_nht_impact = non_carcinogenic_human_toxicity_impact / impact_thresholds["human toxicity: non-carcinogenic"] * penalty_factor
        normalized_fet_impact = freshwater_ecotoxicity_impact / impact_thresholds["ecotoxicity: freshwater"] * penalty_factor
        normalized_ir_impact = ionising_radiation_impact / impact_thresholds["ionising radiation"] * penalty_factor
        normalized_erd_impact = energy_resources_depletion_impact / impact_thresholds["energy resources depletion: non-renewable"] * penalty_factor
        normalized_mr_impact = material_resources_impact / impact_thresholds["material resources: metals/minerals"] * penalty_factor

        print(f"Normalized Climate Change Impact: {normalized_cc_impact}; (penalty = {penalty_factor})")
        print(f"Normalized Ozone Depletion Impact: {normalized_od_impact}; (penalty = {penalty_factor})")
        print(f"Normalized Marine Eutrophication Impact: {normalized_meu_impact}; (penalty = {penalty_factor})")
        print(f"Normalized Freshwater Eutrophication Impact: {normalized_feu_impact}; (penalty = {penalty_factor})")
        print(f"Normalized Terrestrial Acidification Impact: {normalized_tac_impact}; (penalty = {penalty_factor})")
        print(f"Normalized Land Use Impact: {normalized_lu_impact}; (penaliy = {penalty_factor})")
        print(f"Normalized Water Use Impact: {normalized_wu_impact}; (penalty = {penalty_factor})")
        print(f"Normalized Photochemical Ozone Impact: {normalized_pof_impact}; (penalty = {penalty_factor})")
        print(f"Normalized Carcinogenic Human Toxicity Impact: {normalized_cht_impact}; (penalty = {penalty_factor})")
        print(f"Normalized Non-Carcinogenic Human Toxicity Impact: {normalized_nht_impact}; (penalty = {penalty_factor})")
        print(f"Normalized Freshwater Ecotoxicity Impact: {normalized_fet_impact}; (penalty = {penalty_factor})")
        print(f"Normalized Ionising Radiation Impact: {normalized_ir_impact}; (penalty = {penalty_factor})")
        print(f"Normalized Energy Resource Depletion Impact: {normalized_erd_impact}; (penalty = {penalty_factor})")
        print(f"Normalized Material Resources Impact: {normalized_mr_impact}; (penalty = {penalty_factor})")
        
        impact_details = [
            ("Climate Change Impact", climate_change_impact, impact_results_df.loc["climate change"].tolist()),
            ("Ozone Depletion Impact", ozone_depletion_impact, impact_results_df.loc["ozone depletion"].tolist()),
            ("Marine Eutrophication Impact", marine_eutrophication_impact, impact_results_df.loc["eutrophication: marine"].tolist()),
            ("Freshwater Eutrophication Impact", freshwater_eutrophication_impact, impact_results_df.loc["eutrophication: freshwater"].tolist()),
            ("Terrestrial Acidification Impact", terrestrial_acidification_impact, impact_results_df.loc["acidification: terrestrial"].tolist()),
            ("Land Use Impact", land_use_impact, impact_results_df.loc["land use"].tolist()),
            ("Water Use Impact", water_use_impact, impact_results_df.loc["water use"].tolist()),
            ("Photochemical Ozone Formation Impact", photochemical_ozone_formation_impact, impact_results_df.loc["photochemical oxidant formation: human health"].tolist()),
            ("Carcinogenic Human Toxicity Impact", carcinogenic_human_toxicity_impact, impact_results_df.loc["human toxicity: carcinogenic"].tolist()),
            ("Non-Carcinogenic Human Toxicity Impact", non_carcinogenic_human_toxicity_impact, impact_results_df.loc["human toxicity: non-carcinogenic"].tolist()),
            ("Freshwater Ecotoxicity Impact", freshwater_ecotoxicity_impact, impact_results_df.loc["ecotoxicity: freshwater"].tolist()),
            ("Ionising Radiation Impact", ionising_radiation_impact, impact_results_df.loc["ionising radiation"].tolist()),
            ("Energy Resources Depletion Impact", energy_resources_depletion_impact, impact_results_df.loc["energy resources depletion: non-renewable"].tolist()),
            ("Material Resources Impact", material_resources_impact, impact_results_df.loc["material resources: metals/minerals"].tolist())
        ]
        return ((normalized_cc_impact, normalized_od_impact, normalized_meu_impact, normalized_feu_impact, normalized_tac_impact, normalized_lu_impact, normalized_wu_impact, normalized_pof_impact, normalized_cht_impact, normalized_nht_impact, normalized_fet_impact, normalized_ir_impact, normalized_erd_impact, normalized_mr_impact), impact_details, lca_system_model)
    else:
        raise ValueError(f"Target Function {target_function} unknown")    