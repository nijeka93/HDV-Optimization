import math
import random
import pandas as pd
import numpy as np
import os
import json
import multiprocessing as mp
import sys

# --- Debug helpers ------------------------------------------------------------
DEBUG_ARGS_CSV = "debug_call_args_enumeration.csv"
DEBUG_LCA_MODEL_CSV = "debug_lca_model_enumeration.csv"
DEBUG_IMPACTS_CSV = "debug_impacts_enumeration.csv"

def _write_debug_csvs(args_row: dict, lca_model: dict, impact_details):
    # args
    pd.DataFrame([args_row]).to_csv(DEBUG_ARGS_CSV, mode="a", index=False, header=not os.path.exists(DEBUG_ARGS_CSV))
    # lca model
    pd.DataFrame([lca_model]).to_csv(DEBUG_LCA_MODEL_CSV, mode="a", index=False, header=not os.path.exists(DEBUG_LCA_MODEL_CSV))
    # impacts
    # impact_details is either (category, sum, aspects) or list thereof
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

# Import common modules from your project.
from input_excel import load_parameters_from_excel
from prep_optimization_excel import create_design_space
from optimization import objective_function
from data_manager import get_data_point
from itertools import product

# Parallel + checkpoint configuration
OUTPUT_CSV = "enumerated_design_results.csv"
PROGRESS_JSON = "enumeration_progress.json"
CHECKPOINT_EVERY = 50000   # write every N rows
CHUNK_SIZE = 200           # batches submitted to workers
DEFAULT_WORKERS = max(1, mp.cpu_count() - 4)

# Optional: restrict enumeration to a shortlist of (Modules in Series, Modules in Parallel)
# Example: CANDIDATE_PAIRS = [(1, 3)]  # only enumerate 1x3
# Leave empty list [] to enumerate the full grid (1–5 × 1–10)
CANDIDATE_PAIRS = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]  # <- set your shortlist here


# =============================================================================
# 1. Load Shared Parameters from Excel
# =============================================================================
file_path = "Battery_Design_Parameters_FE.xlsx"
(vehicle_scope, battery_scope, carculator_scope, energy_storage, 
 scope, target_function, num_objectives, impact_thresholds, weighting_factors) = load_parameters_from_excel(file_path)


impact_processes = ["glider", "powertrain", "energy storage", "energy chain", "maintenance", "EOL", "road", "direct - non-exhaust", "direct - exhaust"]
module_components = ["cell", "module_voltage_sensor", "module_temperature_sensor", "module_pcb", "module_bms", "module_signal_connector"]
pack_components = ["module", "pack_voltage_sensor", "pack_current_sensor", "pack_pcb", "pack_bms", "pack_signal_connector", "pack_power_connector", "pack_busbar", "pack_relay", "pack_fuse"]


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
        decision_var_columns
        + [f"{prefix}_Summed"]
        + [f"{prefix}_Aspect_{aspect}" for aspect in impact_processes]
        + common_columns
    )
else:
    summed = [f"{prefix}_Summed" for prefix in target_function_map.values()]
    aspects = [f"{prefix}_Aspect_{aspect}" for prefix in target_function_map.values() for aspect in impact_processes]
    all_columns = decision_var_columns + summed + aspects + common_columns

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

# --- Helper: convert a computed cell-count to a safe integer bound ---
def bound_from_cells(cells_value: float, cap: int, floor: int = 1) -> int:
    """Round to nearest int, then clamp to [floor, cap]. If parsing fails, return floor."""
    try:
        v = float(cells_value)
    except Exception:
        return floor
    return max(floor, min(cap, int(round(v))))

# --- Dynamic bounds for enumeration (cap at 10, at least 1) ---
MAX_SERIES   = bound_from_cells(cells_in_series_system, cap=5)
MAX_PARALLEL = bound_from_cells(cells_in_parallel_system, cap=10)
if __name__ == '__main__':
    print(f"Bounds -> Modules in Series: 1..{MAX_SERIES}, Modules in Parallel: 1..{MAX_PARALLEL}")

if __name__ == '__main__':
    print("\n--- Base Design Space ---")
    print(f"Cells per pack (baseline): {cells_per_pack}")
    print(f"Cells in series (baseline): {cells_in_series_system}")
    print(f"Cells in parallel (baseline): {cells_in_parallel_system}")
    print(f"Mass per cell: {mass_per_cell} kg\n")

# =============================================================================
# 3. Full Enumeration through the entire Design Space (Parallel + Resume)
# =============================================================================
# Define ranges for enumeration
modules_in_series_range = range(1, MAX_SERIES + 1)
modules_in_parallel_range = range(1, MAX_PARALLEL + 1)
module_design_fixed = 3  # fixed module design for enumeration

module_component_combinations = list(product([0, 1], repeat=6))
pack_component_combinations = list(product([0, 1], repeat=10))

# Precompute lists for fast index -> tuple mapping
if CANDIDATE_PAIRS:
    pair_list = list(CANDIDATE_PAIRS)
else:
    pair_list = [(mis, mip) for mis in modules_in_series_range for mip in modules_in_parallel_range]

rm_list = module_component_combinations
rp_list = pack_component_combinations

LEN_PAIR = len(pair_list)
LEN_RM = len(rm_list)
LEN_RP = len(rp_list)
TOTAL = LEN_PAIR * LEN_RM * LEN_RP


def index_to_task(idx: int):
    """Map a linear index to (mis, mip, rm, rp)."""
    a = LEN_RM * LEN_RP
    pair_i = idx // a
    rem1 = idx % a
    rm_i = rem1 // LEN_RP
    rp_i = rem1 % LEN_RP

    mis, mip = pair_list[pair_i]
    return (
        mis,
        mip,
        rm_list[rm_i],
        rp_list[rp_i],
    )


def task_generator(start_idx: int, end_idx: int):
    """Yield tasks (idx, mis, mip, rm, rp) from start to end (exclusive)."""
    for i in range(start_idx, end_idx):
        mis, mip, rm, rp = index_to_task(i)
        yield (i, mis, mip, rm, rp)


def evaluate(task):
    """Worker: compute one individual and return a CSV row list."""
    _, mis, mip, rm, rp = task
    modules_per_pack = mis * mip
    cells_per_module = cells_per_pack / modules_per_pack

    norm_fit, impact_details, lca_model = objective_function(
        scope=scope,
        target_function=target_function,
        impact_thresholds=impact_thresholds,
        weighting_factors=weighting_factors,
        vehicle_scope=vehicle_scope,
        carculator_scope=carculator_scope,
        module_design=module_design_fixed,
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
        energy_storage=energy_storage,
    )

    if num_objectives == 1:
        impact_sum = norm_fit if isinstance(norm_fit, float) else norm_fit[0]
        row = [
            mis, mip, module_design_fixed, *rm, *rp,
            impact_sum,
            *impact_details[2],  # aspect list
            lca_model["curb_mass"],
            lca_model["total_battery_mass"],
            lca_model["battery_cell_mass"],
            lca_model["battery_bop_mass"],
            lca_model["battery_lifetime_replacements"],
            lca_model["battery_cell_mass_share"],
            lca_model["battery_cell_mass_share_nmc"],
            lca_model["TtW_energy"],
            lca_model["TtW_efficiency"],
            lca_model["electricity_consumption"],
        ]
    else:
        impact_sums = [impact[1] for impact in impact_details]
        impact_aspects = [v for impact in impact_details for v in impact[2]]
        row = [
            mis, mip, module_design_fixed, *rm, *rp,
            *impact_sums,
            *impact_aspects,
            lca_model["curb_mass"],
            lca_model["total_battery_mass"],
            lca_model["battery_cell_mass"],
            lca_model["battery_bop_mass"],
            lca_model["battery_lifetime_replacements"],
            lca_model["battery_cell_mass_share"],
            lca_model["battery_cell_mass_share_nmc"],
            lca_model["TtW_energy"],
            lca_model["TtW_efficiency"],
            lca_model["electricity_consumption"],
        ]
    return row


def write_header_if_needed():
    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        df_empty = pd.DataFrame(columns=all_columns)
        df_empty.to_csv(OUTPUT_CSV, index=False)


def save_checkpoint(next_index: int):
    with open(PROGRESS_JSON, "w") as f:
        json.dump({"next_index": next_index, "total": TOTAL}, f)


def load_checkpoint():
    if os.path.exists(PROGRESS_JSON):
        try:
            with open(PROGRESS_JSON, "r") as f:
                data = json.load(f)
                return int(data.get("next_index", 0))
        except Exception:
            return 0
    return 0


if __name__ == '__main__':
    mp.freeze_support()

    # Seed an initial progress file so sanity_check.py has data even before first checkpoint
    if not os.path.exists(PROGRESS_JSON):
        with open(PROGRESS_JSON, "w") as f:
            json.dump({"next_index": 0, "total": TOTAL}, f)

    # Determine resume point
    start_index = load_checkpoint()
    end_index = TOTAL

    print(f"Total individuals: {TOTAL}")
    print(f"Resuming at index: {start_index} of {TOTAL}")

    # --- Single-config debug path -------------------------------------------
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

        modules_per_pack = mis * mip
        cells_per_module = cells_per_pack / modules_per_pack

        args_row = {
            "source": "enum-debug",
            "modules_in_series": mis,
            "modules_in_parallel": mip,
            "modules_per_pack": modules_per_pack,
            "cells_per_module": cells_per_module,
            "sum_rm": int(sum(rm)),
            "sum_rp": int(sum(rp)),
            "number_of_packs": int(get_data_point("battery_scope", "number_of_packs")),
            "cells_in_parallel": int(cells_in_parallel_system),
            "module_design": int(module_design_fixed),
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
            module_design=module_design_fixed,
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
            energy_storage=energy_storage,
        )

        _write_debug_csvs(args_row, lca_model, impact_details)
        print("Single debug configuration evaluated via enumeration. Files written:")
        print(f"- {DEBUG_ARGS_CSV}\n- {DEBUG_LCA_MODEL_CSV}\n- {DEBUG_IMPACTS_CSV}")
        raise SystemExit(0)
    else:
        print("[debug] No debug config found; running full enumeration.")

    # Ensure CSV header exists
    write_header_if_needed()

    workers = int(os.environ.get("ENUM_WORKERS", DEFAULT_WORKERS))
    print(f"Using {workers} worker processes.")

    buffer = []
    processed_since_checkpoint = 0
    current_index = start_index

    with mp.get_context("spawn").Pool(processes=workers) as pool:
        for row in pool.imap(evaluate, task_generator(start_index, end_index), chunksize=CHUNK_SIZE):
            buffer.append(row)
            processed_since_checkpoint += 1
            current_index += 1

            if processed_since_checkpoint >= CHECKPOINT_EVERY:
                df_chunk = pd.DataFrame(buffer, columns=all_columns)
                # Append without header
                df_chunk.to_csv(OUTPUT_CSV, mode="a", index=False, header=False)
                buffer.clear()
                processed_since_checkpoint = 0
                save_checkpoint(current_index)
                print(f"Checkpoint saved at index {current_index}/{TOTAL}")

    # Flush remaining
    if buffer:
        df_chunk = pd.DataFrame(buffer, columns=all_columns)
        df_chunk.to_csv(OUTPUT_CSV, mode="a", index=False, header=False)
        buffer.clear()

    save_checkpoint(current_index)
    print("Final checkpoint saved.")
    print(f"Results appended to {OUTPUT_CSV}")
