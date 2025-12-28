import pandas as pd
import torch
import time
import glob
from optimizer_engine import EvolutionaryOptimizer

def run_manual_pareto():
    print("Manual Pareto Analysis (Limit Sweep)...")

    # Sample: MOVOLVO 420 CK001 (Known successful sample)
    # You can get target values manually or from file.
    # Here as example MOVOLVO data (can also get from CSV)
    # Get target_vals from MOVOLVO row in eval_dataset_clean.csv:
    # MOVOLVO 420 CK001 Target Cols (L, a, b at 5 angles):
    # 43.6,30.92,17.02,11.89,9.72, 8.52,7.72,7.1,7.57,7.32, -14.01,-11.75,-8.37,-6.96,-6.41

    # Read from file
    df = pd.read_csv("eval_dataset_clean.csv")
    row = df[df['SampleNo'].str.contains("MOVOLVO", na=False)].iloc[0]
    
    model_files = glob.glob("models/model_fold_*.pt")
    chk = torch.load(model_files[0], map_location="cpu",weights_only=False)
    input_cols = chk['input_cols']
    target_cols = chk['target_cols']
    
    target_vals = row[target_cols].astype(str).str.replace(',', '.').astype(float).values
    
    results = []

    # Try limits from 3 to 8
    for limit in range(3, 9):
        print(f"\nTesting: Maximum {limit} Pigments")

        optimizer = EvolutionaryOptimizer(target_vals, input_cols, target_cols, device="cpu")

        # Set limit
        optimizer.set_constraints(range(len(input_cols)), limit)
        # Increase penalty to not exceed limit (Strict Pareto)
        optimizer.sparsity_limit = limit 
        
        # 1. DE
        pop = optimizer.initialize_population()
        best_de = 999.0
        best_rec = None
        for _ in range(40):  # 40 generations sufficient
            pop, rec, val = optimizer.step_de(pop)
            if val < best_de:
                best_de = val
                best_rec = rec

        # 2. Fine Tune
        final_rec, final_de = optimizer.fine_tune(best_rec, steps=150)

        # How many pigments actually used?
        real_count = (final_rec > 0.001).sum().item()

        print(f"   Result -> DeltaE: {final_de:.4f} | Pigments: {real_count}")
        
        results.append({
            "Constraint_Limit": limit,
            "Real_Pigment_Count": real_count,
            "Delta_E": final_de
        })

    # Save
    pd.DataFrame(results).to_csv("manual_pareto_results.csv", index=False)
    print("\nManual Pareto data ready!")

if __name__ == "__main__":
    run_manual_pareto()