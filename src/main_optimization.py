import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
# --- Force-local imports to win over similarly named modules on PYTHONPATH ---
import os
import json
import sys
_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _LOCAL_DIR not in sys.path[:1]:
    sys.path.insert(0, _LOCAL_DIR)
from deap import base, creator, tools, algorithms
from input_excel import load_parameters_from_excel
from prep_optimization_excel import create_design_space
import optimization as _optmod
print(f"[debug] using optimization module at: {_optmod.__file__}")
objective_function = _optmod.objective_function
from data_manager import get_data_point

# --- Debug helpers ------------------------------------------------------------
DEBUG_ARGS_CSV = "debug_call_args_optimization.csv"
DEBUG_LCA_MODEL_CSV = "debug_lca_model_optimization.csv"
DEBUG_IMPACTS_CSV = "debug_impacts_optimization.csv"

def _write_debug_csvs(args_row: dict, lca_model: dict, impact_details):
    pd.DataFrame([args_row]).to_csv(DEBUG_ARGS_CSV, mode="a", index=False, header=not os.path.exists(DEBUG_ARGS_CSV))
    pd.DataFrame([lca_model]).to_csv(DEBUG_LCA_MODEL_CSV, mode="a", index=False, header=not os.path.exists(DEBUG_LCA_MODEL_CSV))
    if isinstance(impact_details, tuple):
        records = [{"category": impact_details[0], "sum": impact_details[1], **{f"aspect_{i}": v for i, v in enumerate(impact_details[2])}}]
    else:
        records = []
        for cat, s, aspects in impact_details:
            row = {"category": cat, "sum": s}
            row.update({f"aspect_{i}": v for i, v in enumerate(aspects)})
            records.append(row)
    pd.DataFrame(records).to_csv(DEBUG_IMPACTS_CSV, mode="a", index=False, header=not os.path.exists(DEBUG_IMPACTS_CSV))

# --- Debug config loader ------------------------------------------------------
def _load_debug_config():
    """Return (cfg_dict, source_str) if a debug config is provided via env, CLI, or file, else (None, None)."""
    # 1) Environment variable with inline JSON
    dj = os.environ.get("DEBUG_CONFIG_JSON")
    if dj:
        try:
            return json.loads(dj), "env:DEBUG_CONFIG_JSON"
        except Exception as e:
            print(f"[debug] Failed to parse DEBUG_CONFIG_JSON: {e}")
    # 2) Environment variable path to a JSON file
    dp = os.environ.get("DEBUG_CONFIG_PATH")
    if dp and os.path.isfile(dp):
        try:
            with open(dp, "r") as f:
                return json.load(f), f"env:DEBUG_CONFIG_PATH({dp})"
        except Exception as e:
            print(f"[debug] Failed to load DEBUG_CONFIG_PATH {dp}: {e}")
    # 3) CLI flags: --debug-json=<inline json> or --debug-path=<path>
    for a in sys.argv[1:]:
        if a.startswith("--debug-json="):
            payload = a.split("=", 1)[1]
            try:
                return json.loads(payload), "cli:--debug-json"
            except Exception as e:
                print(f"[debug] Failed to parse --debug-json: {e}")
        if a.startswith("--debug-path="):
            p = a.split("=", 1)[1]
            if os.path.isfile(p):
                try:
                    with open(p, "r") as f:
                        return json.load(f), f"cli:--debug-path({p})"
                except Exception as e:
                    print(f"[debug] Failed to load --debug-path {p}: {e}")
    # 4) Fallback file in CWD
    fallback = "DEBUG_CONFIG.json"
    if os.path.isfile(fallback):
        try:
            with open(fallback, "r") as f:
                return json.load(f), f"file:{fallback}"
        except Exception as e:
            print(f"[debug] Failed to load {fallback}: {e}")
    return None, None

# ----- 1. Setup ----------------------------------------------------------------
module_components = ["cell", "module_voltage_sensor", "module_temperature_sensor", "module_pcb", "module_bms", "module_signal_connector"]
pack_components = ["module", "pack_voltage_sensor", "pack_current_sensor", "pack_pcb", "pack_bms", "pack_signal_connector", "pack_power_connector", "pack_busbar", "pack_relay", "pack_fuse"]
best_individual = None


file_path = "data/Battery_Design_Parameters.xlsx"
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

target_energy=get_data_point("battery_scope", "target_energy")
target_voltage=get_data_point("battery_scope", "target_voltage")
number_of_packs=get_data_point("battery_scope", "number_of_packs")

cells_per_pack, cells_in_series_system, cells_in_parallel_system, mass_per_cell = create_design_space(
    target_energy=target_energy,
    target_voltage=target_voltage,
    number_of_packs=number_of_packs
)

# --- Helper: convert a computed cell-count to a safe integer bound ---
def bound_from_cells(cells_value: float, cap: int, floor: int = 1) -> int:
    """Round to nearest int, then clamp to [floor, cap]. If parsing fails, return floor."""
    try:
        v = float(cells_value)
    except Exception:
        return floor
    return max(floor, min(cap, int(round(v))))

# --- Dynamic bounds for module counts (cap at 10, at least 1) ---
MAX_SERIES   = bound_from_cells(cells_in_series_system, cap=5)
MAX_PARALLEL = bound_from_cells(cells_in_parallel_system, cap=10)
print(f"number of packs: {number_of_packs}")
print(f"number of cells in parallel: {cells_in_parallel_system}")
print(f"Bounds -> Modules in Series: 1..{MAX_SERIES}, Modules in Parallel: 1..{MAX_PARALLEL}")

# --- Single-config debug path (bypass GA) -----------------------------------
_cfg, _src = _load_debug_config()
print(f"[debug] DEBUG_CONFIG_JSON present: {bool(os.environ.get('DEBUG_CONFIG_JSON'))}, "
      f"DEBUG_CONFIG_PATH present: {bool(os.environ.get('DEBUG_CONFIG_PATH'))}")
if _cfg:
    print(f"[debug] Entering single-config debug mode (source={_src})")
    try:
        mis = int(_cfg.get("mis"))
        mip = int(_cfg.get("mip"))
    except Exception:
        raise RuntimeError("Debug config must include integer 'mis' and 'mip'")
    rm = _cfg.get("rm", [0]*6)
    rp = _cfg.get("rp", [0]*10)
    if len(rm) != 6 or len(rp) != 10:
        raise RuntimeError("rm must have 6 ints; rp must have 10 ints")
    rm = [int(x) for x in rm]
    rp = [int(x) for x in rp]
    if module_design < 2:
        rm = [0]*6
    modules_per_pack = mis * mip
    cells_per_module = cells_per_pack / modules_per_pack
    args_row = {
        "source": "opt-debug",
        "modules_in_series": mis,
        "modules_in_parallel": mip,
        "modules_per_pack": modules_per_pack,
        "cells_per_module": cells_per_module,
        "sum_rm": int(sum(rm)),
        "sum_rp": int(sum(rp)),
        "number_of_packs": int(get_data_point("battery_scope", "number_of_packs")),
        "cells_in_parallel": int(cells_in_parallel_system),
        "module_design": int(module_design),
        "packing_efficiency": float(get_data_point("battery_scope", "packing_efficiency")),
        "replaceable_mounting_add": float(get_data_point("battery_scope", "replaceable_mounting_add")),
        "scope": str(scope),
        "target_function": str(target_function)
    }
    norm_fit, impact_details, lca_model = objective_function(
        scope=scope,
        target_function=target_function,
        impact_thresholds=impact_thresholds,
        weighting_factors=weighting_factors,
        vehicle_scope=vehicle_scope,
        carculator_scope=carculator_scope,
        module_design=module_design,
        cells_per_module=cells_per_module,
        cells_per_pack=cells_per_pack,
        cells_in_parallel=cells_in_parallel_system,
        modules_per_pack=modules_per_pack,
        modules_in_parallel=mip,
        replaceability_module=rm,
        replaceability_pack=rp,
        replaceable_mounting_add=get_data_point("battery_scope", "replaceable_mounting_add"),
        packing_efficiency=get_data_point("battery_scope", "packing_efficiency"),
        number_of_packs=get_data_point("battery_scope", "number_of_packs"),
        energy_storage=energy_storage
    )
    _write_debug_csvs(args_row, lca_model, impact_details)
    print("Single debug configuration evaluated via optimization. Files written:")
    print(f"- {DEBUG_ARGS_CSV}\n- {DEBUG_LCA_MODEL_CSV}\n- {DEBUG_IMPACTS_CSV}")
    raise SystemExit(0)
else:
    print("[debug] No debug config found; running full GA.")

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
        cells_in_parallel=cells_in_parallel_system,
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
toolbox.register("attr_int_series", lambda: random.randint(1, MAX_SERIES))
toolbox.register("attr_int_parallel", lambda: random.randint(1, MAX_PARALLEL))
toolbox.register("attr_module_component", lambda: 0) # always 0 for module components with a foam filled module design
# toolbox.register("attr_module_component", random.randint, 0, 1)
toolbox.register("attr_pack_component", random.randint, 0, 1)

def create_mixed_individual():
    modules_in_series = random.randint(1, MAX_SERIES)
    modules_in_parallel = random.randint(1, MAX_PARALLEL)
    module_components = [0 for _ in range(6)]
    # module_components = [random.randint(0, 1) for _ in range(6)]
    pack_components = [random.randint(0, 1) for _ in range(10)]
    return creator.Individual([modules_in_series, modules_in_parallel] + module_components + pack_components)

toolbox.register("individual", create_mixed_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", wrapped_objective_function)
toolbox.register("mate", tools.cxTwoPoint)

# --- Exploration tunables ---
INT_MUT_PROB_SERIES = 0.35   # prob to mutate "Modules in Series"
INT_MUT_PROB_PARALLEL = 0.35 # prob to mutate "Modules in Parallel"
BOOL_MUT_PROB = 0.05         # prob to flip a boolean gene

def mutate_mixed(ind):
    # Integer gene 0: Modules in Series (1..MAX_SERIES)
    if random.random() < INT_MUT_PROB_SERIES:
        old = int(ind[0])
        new_val = random.randint(1, MAX_SERIES)
        if new_val == old and MAX_SERIES > 1:
            new_val = 1 + (old % MAX_SERIES)  # ensure a change within bounds
        ind[0] = new_val

    # Integer gene 1: Modules in Parallel (1..MAX_PARALLEL)
    if random.random() < INT_MUT_PROB_PARALLEL:
        old = int(ind[1])
        new_val = random.randint(1, MAX_PARALLEL)
        if new_val == old and MAX_PARALLEL > 1:
            new_val = ((old) % MAX_PARALLEL) + 1  # ensure a change within bounds
        ind[1] = new_val

    # Boolean genes
    for i in range(2, len(ind)):
        # lock module bits to 0 for foam-filled designs
        if module_design < 2 and 2 <= i < 8:
            ind[i] = 0
            continue
        if random.random() < BOOL_MUT_PROB:
            ind[i] = 1 - int(ind[i])

    return (ind,)

toolbox.register("mutate", mutate_mixed)
toolbox.register("select", tools.selNSGA2)

# ----- 4. GA Parameters ------------------------------------------------------
CXPB, MUTPB = 0.8, 0.1
NGEN = 10 # 150
POP_SIZE = 10 # 50
ELITE_SIZE = max(1, round(POP_SIZE / 50))  # At least 1 elite

def generate_balanced_population(pop_size):
    population = []
    for _ in range(pop_size):
        modules_in_series = random.randint(1, MAX_SERIES)
        modules_in_parallel = random.randint(1, MAX_PARALLEL)
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

        # Diversity injection: copy best and jitter integer genes to avoid getting stuck
        injected = creator.Individual(best_individual)
        injected[0] = random.randint(1, MAX_SERIES)
        injected[1] = random.randint(1, MAX_PARALLEL)
        if module_design < 2:
            injected[2:8] = [0]*6

        # Ensure the injected individual has an ID
        assign_id_if_needed(injected)

        # Ensure all individuals currently in new_pop have IDs before filtering
        for ind in new_pop:
            assign_id_if_needed(ind)

        candidates = [ind for ind in new_pop if ind.id_ not in elite_ids]
        if candidates:
            to_remove = random.choice(candidates)
            new_pop.remove(to_remove)
            new_pop.append(injected)

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