# HDV Optimization

This project implements a multi-objective optimization framework for traction battery design for heavy-duty vehicles (HDV), balancing value retention (via replaceable component designs) and system efficiency (via integrated component designs) using a DEAP-based evolutionary algorithm. The battery system design is evaluated via lifecycle assessment (LCA) with the `carculator_truck` model.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone 
cd hdv-optimization
```

### 2. Set Up the Conda Environment

Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.

Then run:

```bash
conda env create -f environment.yaml
conda activate hdv_opt_env
```

If you're using **VS Code**, you can open the folder and select this interpreter via:

- `Ctrl+Shift+P` → `Python: Select Interpreter` → pick `hdv_opt_env`

---

## ⚙️ Configuration

All model parameters and optimization settings are configured via the Excel file:

```bash
Battery_Design_Parameters.xlsx
```

Key inputs include:
- Vehicle and battery design scope
- Environmental impact targets (e.g. Climate Change, Ozone Depletion)
- Number of objectives
- Energy storage targets and voltage levels

---

## 🧬 How the Optimization Works

The core logic is implemented in [`main_optimization.py`](main_optimization.py). Here's a breakdown:

### 🎯 1. Design Space Definition
- Combines user input from the Excel file and calculates:
  - Modules in series & parallel
  - Cell-to-module and module-to-pack structure
  - Replaceability options for components (binary: 0 = fixed, 1 = replaceable)

### 🧠 2. Evolutionary Algorithm Setup (via DEAP)
- Uses **NSGA-II** for multi-objective optimization
- Parameters:
  - Population size: 50
  - Generations: 120
  - Crossover probability: 0.8
  - Mutation probability: 0.1
- Elitism ensures best individuals are preserved each generation

### 🧪 3. Evaluation Function
Each individual is passed to a wrapped objective function that:
- Computes derived parameters (e.g. modules per pack, cells per module)
- Calls the lifecycle assessment (`objective_function`)
- Returns impact metrics + system model data (e.g. battery mass, replacements, efficiency)

### 📊 4. Logging & Output
- Progress is saved every generation to:
  - `optimization_progress.csv` (all generations)
  - `optimization_results.csv` (final archive)
- Includes fitness, impact aspects, and technical metrics (e.g. TtW efficiency)

---

## 📦 Output Data

Your results will be saved in two CSV files:

- `optimization_progress.csv`: appends all individuals from each generation
- `optimization_results.csv`: final snapshot of all individuals and results

Columns include:
- Decision variables (e.g. design, replaceability)
- Environmental metrics (e.g. CC_Summed, OD_Summed)
- Technical results (battery mass, replacements, efficiency, etc.)

---

## 🧩 Dependencies

All dependencies are managed via `environment.yaml`. Key libraries include:
- `deap`
- `numpy`, `pandas`, `matplotlib`
- `carculator_truck` (LCA model)
- Custom modules:
  - `input_excel.py`, `prep_optimization_excel.py`, `optimization.py`, `data_manager.py`

---

## ✅ Status

✔ Functional NSGA-II optimization  
✔ Multi-objective setup  
✔ LCA-based evaluation of traction battery design trade-offs

⬜ Pareto visualization (optional)
⬜ Graphical interface (future)
⬜ Direct manipulation of LCI in carculator_truck instead of passing intermediate parameters (future)
---

## 📬 Questions?

Open an issue or reach out via email for questions, feedback, or collaboration ideas.

---

## 🙏 Acknowledgments

This project builds upon the [carculator_truck](https://github.com/Laboratory-for-Energy-Systems-Analysis/carculator_truck) tool developed by the Technology Assessment group at the Paul Scherrer Institut (PSI). We thank the authors for providing their lifecycle assessment model under the BSD-3-Clause license, which we build upon.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.