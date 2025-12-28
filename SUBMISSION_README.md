# CENG 482 Project Submission

**Project Title:** Hybrid Evolutionary Optimization for Multi-Angle Automotive Color Prediction and Recipe Formulation

**Authors:** Bekir Can Yalçın (290201042), Serhat Uludağ (290201051)


---

## Submission Contents


### 1. Project Report (1 file)
- `CENG482_EA_Course_Project_Final.pdf` - IEEE format report with all experimental results

### 2. Pre-trained Models (5 files)
- `models_production/model_fold_0.pt` - Trained neural network fold 1 (1.2 MB)
- `models_production/model_fold_1.pt` - Trained neural network fold 2 (1.2 MB)
- `models_production/model_fold_2.pt` - Trained neural network fold 3 (1.2 MB)
- `models_production/model_fold_3.pt` - Trained neural network fold 4 (1.2 MB)
- `models_production/model_fold_4.pt` - Trained neural network fold 5 (1.2 MB)

**Note:** These pre-trained models allow immediate testing without retraining (which takes ~30-60 minutes). They are the exact models used to generate the results in the report.

### 3. Source Code (9 files)

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

### 4. Data Files (4 files)
- `RS400_Clean.csv` - Training dataset (2,106 samples, 752KB)
- `eval_dataset_clean.csv` - Test dataset (30 samples, 9.9KB)
- `best_params.txt` - Optimal hyperparameters from Optuna
- `Final_Project_Benchmark_Results_v2.csv` - Experimental results for Table III in report
