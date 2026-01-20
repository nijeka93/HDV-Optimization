from carculator_truck import *
import math
import random
import pandas as pd
import numpy as np

modules_in_series = 1         # Example: 3 modules in series
modules_in_parallel = 6        # Example: 10 modules in parallel
module_design = 3               # 2 => closed non-filled design
module_components = ["cell", "module_voltage_sensor", "module_temperature_sensor", "module_pcb", "module_bms", "module_signal_connector"]
pack_components = ["module", "pack_voltage_sensor", "pack_current_sensor", "pack_pcb", "pack_bms", "pack_signal_connector", "pack_power_connector", "pack_busbar", "pack_relay", "pack_fuse"]

impact_categories = ["climate change", "ozone depletion", "eutrophication: marine", "eutrophication: freshwater", 
                        "acidification: terrestrial", "land use", "water use", "photochemical oxidant formation: human health", 
                        "human toxicity: carcinogenic", "human toxicity: non-carcinogenic", "ecotoxicity: freshwater", 
                        "ionising radiation", "energy resources depletion: non-renewable", "material resources: metals/minerals"]
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
    "Material Resources Impact": "MR"
    }

impact_aspects = ["glider", "powertrain", "energy storage", "energy chain",
                    "maintenance", "EOL", "road", "direct - non-exhaust", "direct - exhaust"]

replaceability_module = [1, 0, 0, 0, 0, 0]
replaceability_pack = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


tip = TruckInputParameters()
tip.static()

new_scope = {
    "powertrain": ["BEV",],
    "size": ["40t"],
    "year": [2030]
}
dcts, array = fill_xarray_from_input_parameters(tip, scope=new_scope)
array.loc[dict(parameter="lifetime kilometers")] = 10000000 # 1,000,000 km

tm = TruckModel(
    array=array,
    country='DE',
    cycle='Long haul', # driving cycles: 'Long haul', 'Urban delivery' & 'Regional delivery'
    target_range = 600, # 600 km
    power={
        ("BEV", "40t", 2030): 340, # 340 kW  
    },
    payload={
        ("BEV", "40t", 2030): 15000, # 15,000 kg
    },
    energy_storage={
        "electric": {
            ("BEV", "40t", 2030): "LFP" # battery chemistries: 'NCA', 'LFP', 'NMC-111', 'NMC-432', 'NMC-622', 'NMC-811', 'NMC-911'
        },
        "origin": "DE"
    }
)

tm.set_all()

    # 5) Adapted values of calculated TruckModel parameters
tm.array.loc[dict(parameter="battery lifetime replacements", value = 0)] = [0.17] # working!!!; positive impact correlation
tm.array.loc[dict(parameter="battery cell mass", value = 0)] = [3396] # working!!!; positive impact correlation
tm.array.loc[dict(parameter="energy battery mass", value = 0)] = [4411] # working!!!; positive impact correlation
tm.array.loc[dict(parameter="battery BoP mass", value = 0)] = [1015] # working!!!; positive impact correlation

# 8) Run LCA calculations to compute environmental impacts and structure results
ic = InventoryTruck(tm, functional_unit="tkm")
res = ic.calculate_impacts()

selected_res = res.sel(impact_category=impact_categories)
impact_results_df = pd.DataFrame(np.squeeze(selected_res.data), index=impact_categories, columns=impact_aspects)

# 9) Process and categorize results
# a) Verify the reshaped dimensions match expectations
if impact_results_df.shape != (len(impact_categories), len(impact_aspects)):
    raise ValueError(f"Mismatch: impact_results_df.shape = {impact_results_df.shape}, expected = ({len(impact_categories)}, {len(impact_aspects)})")
impact_results_df.to_csv("hardcoded_design_results.csv", index=False)
print(f"impact_results_df: {impact_results_df}")
