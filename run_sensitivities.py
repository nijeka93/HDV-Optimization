"""
Run the baseline and/or sensitivity scenarios of main_optimization.py in parallel.

Each scenario runs in its own folder (copy of src/ and data/Battery_Design_Parameters.xlsx),
because main_optimization.py writes optimization_progress.csv / optimization_results.csv
into its working directory (append mode). Scenario changes are passed as parameter
overrides (PARAM_OVERRIDES) and environment variables, so the input Excel stays unchanged.

Usage (from the project root, with the conda environment active):
    python run_sensitivities.py --list
    python run_sensitivities.py --quick                 # 2 generations x 6 individuals, checks all scenarios start
    python run_sensitivities.py                         # all sensitivity scenarios, full GA (150 x 50)
    python run_sensitivities.py --only PL_mix NMC_pendant --jobs 2
    python run_sensitivities.py --out ~/hdv_runs --jobs 6 --seed 42
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Scenario definitions as in the previous submission (Sens_*.xlsx); keys = parameter names
# in column D of 'Input Optimization'. "env" = environment variables for main_optimization.py.
SCENARIOS = {
    "Baseline":               {"overrides": {}},
    "MD0_CC1,3":              {"overrides": {"Module Design": 0, "Cell Clearance": 1.3}},  # foamed closed plastic casing
    "MD1_CC1,7":              {"overrides": {"Module Design": 1, "Cell Clearance": 1.7}},  # foamed open plastic casing
    "MD2_CC1,7":              {"overrides": {"Module Design": 2, "Cell Clearance": 1.7}},  # sealed aluminium casing
    "Cell_clearance_1,8":     {"overrides": {"Cell Clearance": 1.8}},
    "Cell_quality_low":       {"overrides": {"Cell Quality": "low"}},
    "Repl_mass_penalty_0,05": {"overrides": {"Replaceable Mounting: Added Mass": 1.05}},
    "Failure_prop_0,5":       {"overrides": {}, "env": {"FAILURE_PROPAG_PROB": "0.5"}},
    "Signal_connect_0,1x":    {"overrides": {"Module Signal Connector Failure Rate": 0.04389}},
    "Packing_eff_0,8":        {"overrides": {"Volumentric Packing Efficiency": 0.8}},
    "Cell_cycles_5000":       {"overrides": {"Cell Nominal Cycle Life": 5000}},
    "NMC_pendant":            {"overrides": {"Cell Model": "NMC pendant", "Cell Chemistry": "NMC-811",
                                             "Cell Length": 16.7, "Cell Width": 4.1, "Cell Height": 15.6,
                                             "Cell Mass": 2.25, "Cell Gravimetric Energy": 260,
                                             "Cell Nominal Cycle Life": 3000}},
    "Electricity_mix_PL":     {"overrides": {"Electricity Mix Usage": "PL"}},
}

PROJECT = Path(__file__).resolve().parent
INPUT_XLSX = Path("data") / "Battery_Design_Parameters.xlsx"


def prepare_run_dir(run_dir: Path, overwrite: bool) -> None:
    if (run_dir / "optimization_progress.csv").exists() and not overwrite:
        raise FileExistsError(f"{run_dir} already contains results; use --overwrite or another --out folder")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "data").mkdir(parents=True)
    shutil.copytree(PROJECT / "src", run_dir / "src", ignore=shutil.ignore_patterns("__pycache__", "*.csv"))
    shutil.copy2(PROJECT / INPUT_XLSX, run_dir / INPUT_XLSX)


def run_scenario(name: str, cfg: dict, out: Path, args) -> tuple:
    run_dir = out / name
    prepare_run_dir(run_dir, args.overwrite)
    env = os.environ.copy()
    env.update({
        "PARAM_OVERRIDES": json.dumps(cfg.get("overrides", {})),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",  # one core per run
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",  # write run.log live
    })
    env.update(cfg.get("env", {}))
    if args.seed is not None:
        env["GA_SEED"] = str(args.seed)
    if args.quick:
        env.update({"GA_NGEN": "2", "GA_POP_SIZE": "6"})
    (run_dir / "scenario.json").write_text(json.dumps(
        {"name": name, **cfg, "seed": args.seed, "quick": args.quick, "python": sys.executable}, indent=2))
    t0 = time.time()
    with open(run_dir / "run.log", "w") as log:
        rc = subprocess.run([sys.executable, "src/main_optimization.py"], cwd=run_dir, env=env,
                            stdout=log, stderr=subprocess.STDOUT).returncode
    return name, rc, (time.time() - t0) / 60, run_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", help="scenario names to run (default: all except Baseline)")
    p.add_argument("--include-baseline", action="store_true", help="also run the Baseline scenario")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="parallel runs")
    p.add_argument("--out", default="~/hdv_runs", help="output folder (keep it outside iCloud)")
    p.add_argument("--seed", type=int, default=None, help="GA random seed (same seed for all scenarios)")
    p.add_argument("--quick", action="store_true", help="2 generations x 6 individuals, to test the setup")
    p.add_argument("--overwrite", action="store_true", help="replace existing run folders")
    p.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = p.parse_args()

    if args.list:
        for n, c in SCENARIOS.items():
            print(f"{n:24s} {c}")
        return
    # Pre-flight: refuse to start with scripts that lack the scenario settings or the battery fix
    for f, token in [("src/main_optimization.py", "GA_NGEN"), ("src/input_excel.py", "PARAM_OVERRIDES"),
                     ("src/optimization.py", "TruckModelDesignBattery"), ("src/optimization.py", "_bev_fix")]:
        if token not in (PROJECT / f).read_text():
            sys.exit(f"{f} does not contain '{token}' - outdated copy? Aborting.")
    names = args.only or [n for n in SCENARIOS if n != "Baseline" or args.include_baseline]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        sys.exit(f"Unknown scenario(s): {unknown}. Use --list.")
    out = Path(os.path.expanduser(args.out)) / ("quick" if args.quick else "full")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Python: {sys.executable}\nRunning {len(names)} scenario(s) with {args.jobs} parallel job(s) -> {out}")

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_scenario, n, SCENARIOS[n], out, args): n for n in names}
        for f in as_completed(futures):
            name, rc, minutes, run_dir = f.result()
            status = "OK" if rc == 0 and (run_dir / "optimization_results.csv").exists() else f"FAILED (exit {rc})"
            print(f"[{time.strftime('%H:%M')}] {name:24s} {status:16s} {minutes:6.1f} min  -> {run_dir}")
            results.append({"scenario": name, "status": status, "minutes": round(minutes, 1), "folder": str(run_dir)})
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    failed = [r for r in results if r["status"] != "OK"]
    print(f"Done. {len(results) - len(failed)} OK, {len(failed)} failed. See run.log in each folder.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
