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
best_individual = None


file_path = "Battery_Design_Parameters.xlsx"
(
    vehicle_scope,
    battery_scope,
    carculator_scope,
    energy_storage,
    scope,
    target_function,
    num_objectives,
    impact_thresholds,
    weighting_factors,
) = load_parameters_from_excel(file_path)

# Cache module design for reuse
module_design = get_data_point("battery_scope", "module_design")

# Impact and decision columns
impact_processes = ["glider", "powertrain", "energy storage", "energy chain", "maintenance", "EOL", "road", "direct - non-exhaust", "direct - exhaust"]

decision_var_columns = [
    "Modules in Series", 
    "Modules in Parallel", 
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
    replaceability_module = individual[2:8]
    replaceability_pack = individual[8:]

    modules_per_pack = modules_in_series * modules_in_parallel
    print(f"modules per pack: {modules_per_pack}")
    cells_per_module = cells_per_pack / modules_per_pack

    # --- Constraint Check ---
    if module_design < 2 and any(replaceability_module):
        # Penalize: Return a very high penalty for each objective.
        distance_to_target = (1e6,) * num_objectives
        if num_objectives == 1:
            impact_details = ("foam constraint violation", 1e6, [0] * 9)
        else:
            impact_details = [("constraint violation", 1e6, [0] * 9) for _ in range(num_objectives)]
        lca_system_model = {
            "curb_mass": 0,
            "total_battery_mass": 0,
            "battery_cell_mass": 0,
            "battery_bop_mass": 0,
            "battery_lifetime_replacements": 0,
            "battery_cell_mass_share": 0,
            "battery_cell_mass_share_nmc": 0,
            "TtW_energy": 0,
            "TtW_efficiency": 0,
            "electricity_consumption": 0,
            "battery_cycle_life": 0
        }
        return distance_to_target, impact_details, lca_system_model
    # --- End Constraint Check ---

    # Evaluate actual objective function if constraint not violated
    distance_to_target, impact_details, lca_system_model = objective_function(
        scope=scope,
        target_function=target_function,
        impact_thresholds=impact_thresholds,
        weighting_factors=weighting_factors,
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

    # Debug: warn if module bits are nonzero for foam-filled designs
    if module_design < 2 and any(replaceability_module):
        print("[WARN] Module bits non-zero for foam-filled design; they will be penalized unless repaired upstream.")

    return distance_to_target, impact_details, lca_system_model

# ----- 3. DEAP Setup ---------------------------------------------------------
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti

# pylint: disable=all
creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * num_objectives)
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
# register individuals of the initial population
toolbox.register("attr_int_series", random.randint, 1, 5)
toolbox.register("attr_int_parallel", random.randint, 1, 15)
toolbox.register("attr_module_component", lambda: 0) # always 0 for module components with a foam filled module design
# toolbox.register("attr_module_component", random.randint, 0, 1)
toolbox.register("attr_pack_component", random.randint, 0, 1)

def create_mixed_individual():
    modules_in_series = random.randint(1, 5)
    modules_in_parallel = random.randint(1, 15)
    module_components = [0 for _ in range(6)]
    # module_components = [random.randint(0, 1) for _ in range(6)]
    pack_components = [random.randint(0, 1) for _ in range(10)]
    return creator.Individual([modules_in_series, modules_in_parallel] + module_components + pack_components)

toolbox.register("individual", create_mixed_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", wrapped_objective_function)
toolbox.register("mate", tools.cxTwoPoint)

# Build mutation bounds dynamically so module components are locked to 0 for foam-filled designs
if module_design < 2:
    # [Modules in Series, Modules in Parallel] + [6 module bits] + [10 pack bits]
    mut_low = [1, 1] + [0]*6 + [0]*10
    mut_up  = [5, 15] + [0]*6 + [1]*10  # module bits cannot mutate to 1
else:
    mut_low = [1, 1] + [0]*6 + [0]*10
    mut_up  = [5, 15] + [1]*6 + [1]*10

toolbox.register("mutate", tools.mutUniformInt, low=mut_low, up=mut_up, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

# ----- 4. GA Parameters ------------------------------------------------------
CXPB, MUTPB = 0.8, 0.1
NGEN = 150
POP_SIZE = 50
ELITE_SIZE = max(1, round(POP_SIZE / 50))  # At least 1 elite

def generate_balanced_population(pop_size):
    population = []
    for _ in range(pop_size):
        modules_in_series = random.randint(1, 5)
        modules_in_parallel = random.randint(1, 15)
        module_bits = [random.randint(0, 1) for _ in range(6)]
        pack_bits = [random.randint(0, 1) for _ in range(10)]
        individual = creator.Individual([modules_in_series, modules_in_parallel] + module_bits + pack_bits)
        population.append(individual)
    return population

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

    # Repair: for closed, foam-filled module designs, module components must be integrated (0)
    if module_design < 2:
        for ind in offspring:
            ind[2:8] = [0]*6

    # Ensure all offspring have unique IDs before any archive lookups
    for ind in offspring:
        assign_id_if_needed(ind)

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

    # Update best individual across generations
    for ind in offspring:
        if ind.fitness.valid:
            if (best_individual is None) or (ind.fitness.values < best_individual.fitness.values):
                best_individual = creator.Individual(ind)
                best_individual.fitness.values = ind.fitness.values

    # --- Instead of separate logging for invalid individuals and elites,
    #      log the entire population of this generation.
    for ind in pop:
        assign_id_if_needed(ind)
    top_ind = tools.selBest(offspring, ELITE_SIZE)

    # Remove the id_ attribute from elites so that new IDs are generated
    for e in top_ind:
        if hasattr(e, 'id_'):
            del e.id_

    # Now, clone elites for strict elitism:
    elites = [creator.Individual(e) for e in top_ind]
    # Re-assign new IDs to these clones:
    for e in elites:
        assign_id_if_needed(e)
    for e in elites:
        if e.id_ in archive:
            imp_det, lca_sys_model = archive[e.id_]
            if num_objectives == 1:
                distance_to_target = imp_det[1]
            else:
                distance_to_target = tuple(impact[1] for impact in imp_det)
            e.fitness.values = distance_to_target if isinstance(distance_to_target, tuple) else (distance_to_target,)

    # Select the rest from offspring (fill up to POP_SIZE - ELITE_SIZE)
    remainder = toolbox.select(offspring, len(pop) - ELITE_SIZE)

    # New population is elites + remainder
    new_pop = elites + remainder

    # Enforce one elite (global best across generations)
    elite_ids = {ind.id_ for ind in elites}
    if best_individual is not None:
        elite = creator.Individual(best_individual)
        assign_id_if_needed(elite)
        if all((not hasattr(ind, 'id_')) or (elite.id_ != ind.id_) for ind in new_pop):
            non_elite_inds = [ind for ind in new_pop if (not hasattr(ind, 'id_')) or (ind.id_ not in elite_ids)]
            if non_elite_inds:
                to_remove = random.choice(non_elite_inds)
                new_pop.remove(to_remove)
                new_pop.append(elite)

    # Ensure all individuals in the new population have IDs
    for ind in new_pop:
        assign_id_if_needed(ind)

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
                ind[0], ind[1],
                *ind[2:8], *ind[8:],
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
                ind[0], ind[1],
                *ind[2:8], *ind[8:],
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
    environmental_impact_all_individuals.append([ind.fitness.values for ind in new_pop if ind.fitness.valid])
    print(f"Generation {generation} completed. Population logged: {len(new_pop)} individuals.")

    # Proceed with selection for the next generation
    # (No extra logging or separate elite loop now; every individual is logged in the unified loop.)
    pop = new_pop

print("All generations done. Saving final results...")
df_final = pd.DataFrame(individuals_data, columns=all_columns)
df_final.to_csv("optimization_results.csv", index=False)
print("Saved final results to optimization_results.csv")