# CENG 482 Project Submission

**Project Title:** Hybrid Evolutionary Optimization for Multi-Angle Automotive Color Prediction and Recipe Formulation

**Authors:** Bekir Can Yalçın (290201042), Serhat Uludağ (290201051)

**Institution:** Izmir Institute of Technology

**Due Date:** December 31, 2025

---

## Submission Contents

This submission includes **14 files total:**

### 1. Project Report (1 file)
- `CENG482_EA_Course_Project_Final.pdf` - IEEE format report with all experimental results

### 2. Source Code (9 files)

#### Phase 1: Forward Model Training & Optimization
- `train_production.py` - Train 5-fold ensemble neural network
- `optimize_model.py` - Hyperparameter optimization using Differential Evolution
- `tune.py` - Hyperparameter optimization using Optuna (TPE)

#### Phase 2: Evolutionary Algorithms
- `optimizer_engine.py` - Core EA implementations (GA, DE, PSO) + gradient fine-tuning
- `benchmark.py` - Algorithm comparison (GA vs DE vs PSO)
- `benchmark_full.py` - Production hybrid system (DE + Gradient Descent)

#### Phase 3: Multi-Objective Optimization
- `nsga2_engine.py` - NSGA-II implementation for Pareto optimization
- `run_pareto.py` - Generate Pareto frontier (accuracy vs cost)
- `plot_pareto.py` - Visualization of cost-accuracy tradeoff

### 3. Data Files (4 files)
- `RS400_Clean.csv` - Training dataset (2,106 samples, 752KB)
- `eval_dataset_clean.csv` - Test dataset (30 samples, 9.9KB)
- `best_params.txt` - Optimal hyperparameters from Optuna
- `Final_Project_Benchmark_Results_v2.csv` - Experimental results for Table III in report

