import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from deap import base, creator, tools, algorithms
from input_excel import load_parameters_from_excel
from prep_optimization_excel import create_design_space
from optimization import objective_function
from data_manager import get_data_point

# ----- 1. Setup ----------------------------------------------------------------
module_components = ["cell", "module_voltage_sensor", "module_temperature_sensor", "module_pcb", "module_bms", "module_signal_connector"]
pack_components = ["module", "pack_voltage_sensor", "pack_current_sensor", "pack_pcb", "pack_bms", "pack_signal_connector", "pack_power_connector", "pack_busbar", "pack_relay", "pack_fuse"]

file_path = "Battery_Design_Parameters.xlsx"
(vehicle_scope, battery_scope, carculator_scope, energy_storage, 
 scope, target_function, num_objectives, impact_thresholds) = load_parameters_from_excel(file_path)

# Impact and decision columns
impact_processes = ["glider", "powertrain", "energy storage", "energy chain", "maintenance", "EOL", "road", "direct - non-exhaust", "direct - exhaust"]

decision_var_columns = [
    "Modules in Series", 
    "Modules in Parallel", 
    "Module Design"
] + module_components + pack_components

target_function_map = {
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
    "Energy Resource Depletion Impact": "ERD",
    "Material Resource Impact": "MR",
}

common_columns = [
    "curb_mass", "total_battery_mass", "battery_cell_mass", "battery_bop_mass",
    "battery_lifetime_replacements", "battery_cell_mass_share", "battery_cell_mass_share_NMC", 
    "TtW_energy", "TtW_efficiency", "electricity_consumption"
]

if target_function in target_function_map:
    prefix = target_function_map[target_function]
    all_columns = (
        ["Generation", "Is_Elite"]
        + decision_var_columns
        + [f"{prefix}_Summed"]
        + [f"{prefix}_Aspect_{aspect}" for aspect in impact_processes]
        + common_columns
    )
else:
    # Multi-objective (include all)
    summed = [f"{prefix}_Summed" for prefix in target_function_map.values()]
    aspects = [f"{prefix}_Aspect_{aspect}" for prefix in target_function_map.values() for aspect in impact_processes]
    all_columns = (
        ["Generation", "Is_Elite"]
        + decision_var_columns
        + summed
        + aspects
        + common_columns
    )

cells_per_pack, cells_in_series, cells_in_parallel, mass_per_cell = create_design_space(
    target_energy=get_data_point("battery_scope", "target_energy"),
    target_voltage=get_data_point("battery_scope", "target_voltage"),
    number_of_packs=get_data_point("battery_scope", "number_of_packs")
)

# ----- 2. Objective / Fitness Wrapper ----------------------------------------
def wrapped_objective_function(individual):
    modules_in_series = individual[0]
    modules_in_parallel = individual[1]
    module_design = individual[2]
    replaceability_module = individual[3:9]
    replaceability_pack = individual[9:]
    
    # # --- Constraint Check: ---
    # # For foam-filled designs (module_design 0 or 1), none of the module components should be replaceable.
    # if module_design < 2 and any(replaceability_module):
    #     # Penalize: Return a very high penalty for each objective.
    #     penalty = (1e6,) * num_objectives
    #     # Create a dummy impact detail tuple. For single-objective, assume a 3-tuple:
    #     dummy_details = ("foam constraint violation", 1e6, [0] * 9)
    #     # Create a dummy lca_system_model dictionary (or set default values)
    #     dummy_lca = {
    #         "curb_mass": 0,
    #         "total_battery_mass": 0,
    #         "battery_cell_mass": 0,
    #         "battery_bop_mass": 0,
    #         "battery_lifetime_replacements": 0,
    #         "battery_cell_mass_share": 0,
    #         "battery_cell_mass_share_nmc": 0,
    #         "TtW_energy": 0,
    #         "TtW_efficiency": 0,
    #         "electricity_consumption": 0
    #     }

    #     return penalty, dummy_details, dummy_lca
    #     # --- End Constraint Check ---
    
    # Derived calculations using GA genes rather than create_design_space results
    # (Adjust if necessary to match your desired realistic ranges.)
    modules_per_pack = modules_in_series * modules_in_parallel
    print(f"modules per pack: {modules_per_pack}")
    cells_per_module = cells_per_pack / modules_per_pack
    
    distance_to_target, impact_details, lca_system_model = objective_function(
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
    
    return distance_to_target, impact_details, lca_system_model

# ----- 3. DEAP Setup ---------------------------------------------------------
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti

# pylint: disable=all
creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * num_objectives)
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
# register individuals of the initial population; starting with foam filled module designs only, 
# to ensure proper exploration of the design space
toolbox.register("attr_int_series", random.randint, 1, 5)
toolbox.register("attr_int_parallel", random.randint, 1, 15)
toolbox.register("attr_int_design", random.randint, 0, 1) # start with foam filled module designs only
toolbox.register("attr_module_component", lambda: 0) # always 0 for module components with a foam filled module design
toolbox.register("attr_pack_component", random.randint, 0, 1)

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_int_series, toolbox.attr_int_parallel, toolbox.attr_int_design) + 
                 (toolbox.attr_module_component,) * 6 + 
                 (toolbox.attr_pack_component,) * 10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", wrapped_objective_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, 
                 low=[1, 1, 0] + [0] * 16, 
                 up=[5, 15, 4] + [1] * 16, 
                 indpb=0.05)
toolbox.register("select", tools.selNSGA2)

# ----- 4. GA Parameters ------------------------------------------------------
CXPB, MUTPB = 0.8, 0.1
NGEN = 10
POP_SIZE = 10
ELITE_SIZE = max(1, round(POP_SIZE / 50))  # At least 1 elite

pop = toolbox.population(n=POP_SIZE)

# ----- 5. Tracking / Archive Setup -------------------------------------------
# We'll assign each individual a unique id_ to store its evaluated data in an archive.
environmental_impact_all_individuals = []
individuals_data = []
id_counter = itertools.count()
archive = {}  # archive[ind.id_] = (battery_replacements, cell_to_pack_mass_share, impact_details)

def assign_id_if_needed(ind):
    if not hasattr(ind, 'id_'):
        ind.id_ = next(id_counter)

# ----- 6. Main Evolutionary Loop ---------------------------------------------
for generation in range(NGEN):
    # Variation: apply crossover and mutation
    offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
    
    # Evaluate any offspring without valid fitness (if any)
    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_individuals:
        assign_id_if_needed(ind)
    fits = list(map(toolbox.evaluate, invalid_individuals))
    
    # Log newly evaluated offspring into archive:
    for ind, fit in zip(invalid_individuals, fits):
        distance_to_target, impact_details, lca_system_model = fit
        # Ensure distance_to_target is a tuple:
        normalized_fit = distance_to_target if isinstance(distance_to_target, tuple) else (distance_to_target,)
        ind.fitness.values = normalized_fit
        archive[ind.id_] = (impact_details, lca_system_model)
    
    # --- Instead of separate logging for invalid individuals and elites,
    #      log the entire population of this generation.
    for ind in pop:
        assign_id_if_needed(ind)
    top_ind = tools.selBest(pop, ELITE_SIZE)
    
    # Remove the id_ attribute from elites so that new IDs are generated
    for e in top_ind:
        if hasattr(e, 'id_'):
            del e.id_
            
    # Now, clone elites for strict elitism:
    elites = [creator.Individual(e) for e in top_ind]
    # Re-assign new IDs to these clones:
    for e in elites:
        assign_id_if_needed(e)
         
    # Select the rest from offspring (fill up to POP_SIZE - ELITE_SIZE)
    remainder = toolbox.select(offspring, len(pop) - ELITE_SIZE)
    
    # New population is elites + remainder
    new_pop = elites + remainder
    
    elite_ids = {ind.id_ for ind in elites}
    
    generation_data = []
    for ind in new_pop:
        # Retrieve evaluation data from archive (or re-evaluate if missing)
        if ind.id_ in archive:
            imp_det, lca_sys_model = archive[ind.id_]
        else:
            norm_fit, imp_det, lca_sys_model = toolbox.evaluate(ind)
            ind.fitness.values = norm_fit if isinstance(norm_fit, tuple) else (norm_fit,)
            archive[ind.id_] = (imp_det, lca_sys_model)
        
        is_elite = (ind.id_ in elite_ids)
        if num_objectives == 1:
            # For single-objective: impact_details ("impact category", sum, aspect_list)
            individual_data = [
                generation,
                is_elite,
                ind[0], ind[1], ind[2],
                *ind[3:9], *ind[9:],
                imp_det[1],
                *imp_det[2],
                lca_sys_model["curb_mass"],
                lca_sys_model["total_battery_mass"],
                lca_sys_model["battery_cell_mass"],
                lca_sys_model["battery_bop_mass"],
                lca_sys_model["battery_lifetime_replacements"],
                lca_sys_model["battery_cell_mass_share"],
                lca_sys_model["battery_cell_mass_share_nmc"],
                lca_sys_model["TtW_energy"],
                lca_sys_model["TtW_efficiency"],
                lca_sys_model["electricity_consumption"]]
        else:
            # For multi-objective: impact_details = [("Climate Change", sum, [aspects]), ("Ozone Depletion", sum, [aspects])]
            impact_sums = [impact[1] for impact in imp_det]
            impact_aspects = [val for impact in imp_det for val in impact[2]]

            individual_data = [
                generation,
                is_elite,
                ind[0], ind[1], ind[2],
                *ind[3:9], *ind[9:],
                *impact_sums,
                *impact_aspects,
                lca_sys_model["curb_mass"],
                lca_sys_model["total_battery_mass"],
                lca_sys_model["battery_cell_mass"],
                lca_sys_model["battery_bop_mass"],
                lca_sys_model["battery_lifetime_replacements"],
                lca_sys_model["battery_cell_mass_share"],
                lca_sys_model["battery_cell_mass_share_nmc"],
                lca_sys_model["TtW_energy"],
                lca_sys_model["TtW_efficiency"],
                lca_sys_model["electricity_consumption"]
            ]
        generation_data.append(individual_data)
    
    # For each generation, log the entire population (all individuals) regardless of duplication.
    individuals_data.extend(generation_data)
    df_gen = pd.DataFrame(generation_data, columns=all_columns)
    df_gen.to_csv("optimization_progress.csv", mode="a", index=False, header=(generation == 0))
    
    # Track fitness for convergence plots (if desired)
    environmental_impact_all_individuals.append([ind.fitness.values for ind in pop if ind.fitness.valid])
    print(f"Generation {generation} completed. Population logged: {len(pop)} individuals.")
    
    # Proceed with selection for the next generation
    # (No extra logging or separate elite loop now; every individual is logged in the unified loop.)
    pop = new_pop

print("All generations done. Saving final results...")
df_final = pd.DataFrame(individuals_data, columns=all_columns)
df_final.to_csv("optimization_results.csv", index=False)
print("Saved final results to optimization_results.csv")