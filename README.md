# HDV Optimization

This project implements a multi-objective optimization framework for traction battery design for heavy-duty vehicles (HDV), balancing value retention (via replaceable component designs) and system efficiency (via integrated component designs) using a DEAP-based evolutionary algorithm. The battery system design is evaluated via lifecycle assessment (LCA) with the `carculator_truck` model.

The code accompanies the manuscript *Environmental life-cycle optimization of traction battery architecture – Balancing structural efficiency and lifetime preservation* (Cleaner Engineering and Technology, CLET-D-26-00170). The version used for revision R3 is described in the section [Revision R3: EF 3.1 correction](#-revision-r3-ef-31-correction).

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/nijeka93/HDV-Optimization.git
cd HDV-Optimization
```

### 2. Set Up the Python Environment

Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.

**Exact environment of the published runs (recommended):** `requirements-lock.txt` pins all package versions used for the results of revision R3 (Python 3.12, `carculator_truck` 0.5.0, `carculator_utils` 1.3.4, `deap` 1.4.3).

```bash
conda create -n hdv_optimization python=3.12
conda activate hdv_optimization
python -m pip install -r requirements-lock.txt
```

**Unpinned environment:** alternatively, `environment.yml` creates an environment with the same name but without fixed versions:

```bash
conda env create -f environment.yml
conda activate hdv_optimization
```

If you're using **VS Code**, open the folder and select the interpreter via:

- `Ctrl+Shift+P` → `Python: Select Interpreter` → pick `hdv_optimization`

On Apple silicon, `scipy` versions below 1.16 may fail to load (`_spropack` dlopen error); upgrading `scipy` to ≥ 1.16 solves this and did not change the results in our test runs.

---

## ⚙️ Configuration

All model parameters and optimization settings are configured via the Excel input files in `data/`:

| File | Used by | Purpose |
|---|---|---|
| `data/Battery_Design_Parameters.xlsx` | `src/main_optimization.py`, `run_sensitivities.py` | Baseline scenario of the optimization |
| `data/Battery_Design_Parameters_Assess.xlsx` | `src/main_assess.py` | Assessment of single, hard-coded designs (e.g. reference battery, optimum) |
| `data/Battery_Design_Parameters_FE.xlsx` | `src/main_full_enumeration.py` | Full enumeration of the baseline design space |

Key inputs include:
- Vehicle and battery design scope (production year, lifetime, payload, driving cycle, electricity mix)
- Cell, module and pack component data (dimensions, masses, failure rates, replaceable-mounting mass penalty)
- Absolute impact thresholds per tkm (planetary boundaries of Sala et al., 2020, down-scaled to Austrian road freight) and EF 3.1 weighting factors (sheet `Absolute Thresholds`)

---

## 🧬 How the Optimization Works

The core logic is implemented in [`src/main_optimization.py`](src/main_optimization.py). Here's a breakdown:

### 🎯 1. Design Space Definition
- Combines user input from the Excel file and calculates:
  - Modules in series & parallel
  - Cell-to-module and module-to-pack structure
  - Replaceability options for components (binary: 0 = integrated, 1 = replaceable)

### 🧠 2. Evolutionary Algorithm Setup (via DEAP)
- Uses **NSGA-II** for multi-objective optimization, with one objective per impact category (impact divided by its absolute threshold)
- Parameters:
  - Population size: 50
  - Generations: 150
  - Crossover probability: 0.8
  - Mutation probability per individual: 0.1 (modules in series/parallel: 0.35 each; flip probability per binary gene: 0.05)
- Elitism: the best individual in terms of the total environmental impact (TEI = sum of normalized impacts × EF 3.1 weighting factors) is carried forward each generation
- Diversity injection: perturbed variants of the module layout of the best individual

### 🧪 3. Evaluation Function
Each individual is passed to a wrapped objective function that:
- Computes derived parameters (e.g. modules per pack, cells per module)
- Calculates the two intermediate metrics: cell-to-pack mass ratio (f1) and battery-equivalent replacements over the vehicle lifetime (f2)
- Calls the life cycle assessment (`objective_function` in `src/optimization.py`) with `carculator_truck`
- Returns impact metrics + system model data (e.g. battery mass, replacements, energy consumption)

### 📊 4. Logging & Output
- Progress is saved every generation to:
  - `optimization_progress.csv` (all generations)
  - `optimization_results.csv` (final archive)
- Includes fitness, impact aspects (life-cycle stages), and technical metrics

---

## ▶️ Running the Scripts

All scripts are run from the project root:

```bash
python src/main_optimization.py      # baseline optimization (150 generations x 50 individuals)
python src/main_assess.py            # evaluate the design hard-coded at the top of the script (module_bits, pack_bits, modules in series/parallel)
python src/main_full_enumeration.py  # full enumeration of the baseline design space (327,680 designs; resumable)
```

`src/main_optimization.py` reads the following optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `GA_SEED` | not set | Random seed for reproducible runs (revision R3: 42) |
| `GA_NGEN`, `GA_POP_SIZE` | 150, 50 | Number of generations and population size (reduce only for test runs) |
| `FAILURE_PROPAG_PROB` | 0.0 | Probability that a cell failure propagates (sensitivity scenario: 0.5) |
| `PARAM_OVERRIDES` | not set | JSON dictionary overriding parameters of the input file by name (used by `run_sensitivities.py`) |

`src/main_full_enumeration.py` writes `enumerated_design_results.csv` and the checkpoint `enumeration_progress.json`; the number of parallel workers can be set with `ENUM_WORKERS`.

### Sensitivity analysis

`run_sensitivities.py` runs the baseline and the 12 sensitivity scenarios of the manuscript in parallel. Each scenario runs in its own folder (copy of `src/` and the input file), and scenario changes are passed as parameter overrides, so the input file stays unchanged.

```bash
python run_sensitivities.py --list                                      # show all scenarios
python run_sensitivities.py --quick --include-baseline                  # test: 2 generations x 6 individuals
python run_sensitivities.py --include-baseline --seed 42 --jobs 12 --out ~/hdv_runs   # full runs (revision R3)
python run_sensitivities.py --only Electricity_mix_PL NMC_pendant      # selected scenarios
```

Keep the output folder outside cloud-synchronized folders (e.g. iCloud), because the runs append to CSV files continuously.

Note: in `carculator_utils` 1.3.4, the battery production location setting has no effect on the inventory; battery manufacturing is represented by the default LFP battery inventory of `carculator_truck`.

---

## 📦 Output Data

The optimization results are saved in two CSV files:

- `optimization_progress.csv`: appends all individuals from each generation
- `optimization_results.csv`: final snapshot of all individuals and results

Columns include:
- Decision variables (modules in series/parallel, replaceability of each component)
- Environmental impacts per tkm, total (e.g. `CC_Summed`, `PM_Summed`) and per life-cycle stage (e.g. `CC_Aspect_energy storage`)
- Technical results (battery mass, cell-to-pack mass ratio, battery-equivalent replacements, curb mass, electricity consumption)

The TEI is calculated after the optimization from the impacts, thresholds and weighting factors. The processed results of revision R3 (all runs, sensitivity analysis, full enumeration) are provided in the Supplementary Information and the associated data repository.

---

## 🧩 Dependencies

Exact versions: `requirements-lock.txt`; unpinned: `environment.yml`. Key libraries include:
- `deap`
- `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`
- `carculator_truck` and `carculator_utils` (LCA model)
- Custom modules in `src/`:
  - `input_excel.py`, `prep_optimization_excel.py`, `optimization.py`, `data_manager.py`, `classes.py`

---

## ✅ Status

✔ Functional NSGA-II optimization  
✔ Multi-objective setup with EF 3.1 impact categories and absolute thresholds  
✔ LCA-based evaluation of traction battery design trade-offs  
✔ Parallel sensitivity runs and full enumeration of the baseline design space

⬜ Pareto visualization (optional)  
⬜ Graphical interface (future)  
⬜ Direct manipulation of LCI in carculator_truck instead of passing intermediate parameters (future)

---

## 📚 References

Sala, S., Crenna, E., Secchi, M., & Sanyé-Mengual, E. (2020). Environmental sustainability of European production and consumption assessed against planetary boundaries. *Journal of Environmental Management*, *269*, 110686. https://doi.org/10.1016/j.jenvman.2020.110686

---

## 📬 Questions?

Open an issue or reach out via email for questions, feedback, or collaboration ideas.

---

## 🙏 Acknowledgments

This project builds upon the [carculator_truck](https://github.com/Laboratory-for-Energy-Systems-Analysis/carculator_truck) tool developed by the Technology Assessment group at the Paul Scherrer Institut (PSI). We thank the authors for providing their lifecycle assessment model under the BSD-3-Clause license, which we build upon.

---

## 📄 License

This project is licensed under the BSD-3 License. See the [LICENSE](LICENSE) file for details.
