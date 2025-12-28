import pandas as pd
import numpy as np
import torch
import time
from optimizer_engine import EvolutionaryOptimizer

def run_benchmark():
    # Configuration
    TEST_FILE = "eval_dataset_clean.csv"
    MODELS_DIR = "models/"

    print("Starting benchmark...")

    # 1. Load dataset
    try:
        df = pd.read_csv(TEST_FILE)
    except Exception as e:
        print(f"Error: Could not read {TEST_FILE}. Check file path. ({e})")
        return

    # 2. Get column names from model checkpoint
    dummy_target = np.zeros(15)
    
    model_files = glob.glob(MODELS_DIR + "model_fold_*.pt")
    if not model_files:
        print("Error: Model files not found! Did you run train_ensemble.py?")
        return

    chk = torch.load(model_files[0], map_location="cpu", weights_only=False)
    input_cols = chk['input_cols']
    target_cols = chk['target_cols']

    print(f"Models found. {len(input_cols)} pigment inputs.")

    # 3. Select 3 random test samples for benchmarking
    samples = df.sample(3)
    results = []
    
    strategies = ["GA", "DE", "PSO"]

    # Main benchmark loop
    for idx, row in samples.iterrows():
        sample_name = str(row.get('SampleNo', f"Sample_{idx}"))
        print(f"\nAnalyzing sample: {sample_name}")

        # Get target L*a*b* values
        try:
            target_vals = row[target_cols].astype(str).str.replace(',', '.').astype(float).values
        except KeyError:
            print(f"Column error: {target_cols} not found in data.")
            continue

        for strat in strategies:
            print(f"  Running {strat}...", end="")
            start_time = time.time()

            # Setup optimizer
            optimizer = EvolutionaryOptimizer(target_vals, input_cols, target_cols, device="cpu")
            optimizer.set_constraints(range(len(input_cols)), 5)  # Max 5 pigments
            
            pop = optimizer.initialize_population()
            
            if strat == "PSO": 
                optimizer.init_pso()
            
            best_de_history = []
            final_de = 999.0

            # Evolution loop (100 generations)
            for gen in range(100):
                if strat == "GA":
                    pop, best_rec, final_de = optimizer.step_ga(pop)
                elif strat == "DE":
                    pop, best_rec, final_de = optimizer.step_de(pop)
                elif strat == "PSO":
                    pop, best_rec, score = optimizer.step_pso(pop)
                    # PSO returns score with penalty, recalculate pure Delta E
                    _, final_de_tensor = optimizer.calculate_fitness_batch(best_rec.unsqueeze(0))
                    final_de = final_de_tensor.item()

                best_de_history.append(final_de)

                # Early stopping
                if final_de < 0.3:
                    break
            
            duration = time.time() - start_time
            print(f" Done. Time: {duration:.2f}s | Delta E: {final_de:.4f}")

            # Count active pigments
            active_pigments = (best_rec > 0.001).sum().item()

            results.append({
                "Sample": sample_name,
                "Algorithm": strat,
                "Final_DeltaE": final_de,
                "Time_Sec": duration,
                "Generations": len(best_de_history),
                "Active_Pigments": active_pigments,
                "History": str(best_de_history)
            })

    # Save results
    res_df = pd.DataFrame(results)
    res_df.to_csv("benchmark_results_cpu.csv", index=False)
    print("\nBenchmark completed! Results saved to 'benchmark_results_cpu.csv'.")

if __name__ == "__main__":
    import glob
    run_benchmark()