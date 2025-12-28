import torch
import numpy as np
from optimizer_engine import EvolutionaryOptimizer

class NSGA2Optimizer(EvolutionaryOptimizer):
    def __init__(self, target_lab, input_cols, target_cols, device="cpu"):
        super().__init__(target_lab, input_cols, target_cols, device)
        # NSGA-II settings
        self.POP_SIZE = 200  # Larger population for Pareto optimization
        
    def calculate_objectives(self, population):
        """
        Returns 2 objectives for NSGA-II:
        1. Delta E (Minimize)
        2. Pigment Count (Minimize)
        """
        # No gradient needed, only sorting
        with torch.no_grad():
            preds = self.predict_batch(population)
            diff = preds.view(-1, 5, 3) - self.target_lab.view(1, 5, 3)

            # Objective 1: Delta E
            delta_e = torch.sqrt(torch.sum(diff**2, dim=2)).mean(dim=1)

            # Objective 2: Active Pigment Count (Soft Count)
            # Count values greater than 0.001
            # Direct counting since it's a non-differentiable operation
            active_counts = (population > 0.001).sum(dim=1).float()

            # Combine two objectives: (N, 2)
            objectives = torch.stack((delta_e, active_counts), dim=1)
            
        return objectives

    def fast_non_dominated_sort(self, objectives):
        """
        Divides population into 'Front' layers.
        Front 0: Best individuals (non-dominated)
        """
        N = objectives.shape[0]
        dominates = (objectives.unsqueeze(1) < objectives.unsqueeze(0)).all(dim=2)
        # Does a dominate b? (a < b in all objectives)

        # Simplified sorting (accelerated with PyTorch)
        ranks = torch.zeros(N, dtype=torch.long, device=self.device)
        current_front = []

        # How many individuals dominate each individual?
        domination_counts = (objectives.unsqueeze(0) > objectives.unsqueeze(1)).all(dim=2).sum(dim=1)

        # Individuals with count 0 are in first front
        current_front = torch.where(domination_counts == 0)[0].tolist()
        
        fronts = [current_front]
        visited = torch.zeros(N, dtype=torch.bool, device=self.device)
        visited[current_front] = True

        # Find other fronts (Simple logic: Remove front, get best of remaining)
        while True:
            # Not yet assigned
            remaining_indices = torch.where(~visited)[0]
            if len(remaining_indices) == 0:
                break

            # Pareto calculation among remaining
            sub_objs = objectives[remaining_indices]
            # Domination count among remaining
            sub_dom_counts = (sub_objs.unsqueeze(0) > sub_objs.unsqueeze(1)).all(dim=2).sum(dim=1)
            
            next_front_local = torch.where(sub_dom_counts == 0)[0]
            next_front_global = remaining_indices[next_front_local]
            
            fronts.append(next_front_global.tolist())
            visited[next_front_global] = True
            
        return fronts

    def crowding_distance_assignment(self, objectives, front_indices):
        """
        Distance calculation to preserve diversity within the same front.
        """
        l = len(front_indices)
        if l == 0: return torch.tensor([], device=self.device)

        distances = torch.zeros(l, device=self.device)
        front_objs = objectives[front_indices]

        # Sort and measure distance for each objective separately
        for m in range(2):  # We have 2 objectives
            sorted_idx = torch.argsort(front_objs[:, m])
            sorted_objs = front_objs[sorted_idx, m]

            # Endpoints at infinite distance (so they're never removed)
            distances[sorted_idx[0]] = 1e9
            distances[sorted_idx[-1]] = 1e9

            # Distance for intermediate points: (Next - Previous) / (Max - Min)
            diff = (sorted_objs[2:] - sorted_objs[:-2])
            scale = sorted_objs[-1] - sorted_objs[0]
            if scale == 0: scale = 1.0
            
            distances[sorted_idx[1:-1]] += diff / scale
            
        return distances

    def evolve_nsga2(self, population):
        # 1. Generate offspring (Crossover & Mutation) - like classic GA
        offspring_pop, _, _ = self.step_ga(population)
        # step_ga uses elitism but we want the full pool, so a bit hacky:
        # In NSGA-II, parent + offspring pools merge (R + Q)

        combined_pop = torch.cat((population, offspring_pop), dim=0)
        # Apply constraints
        combined_pop = self.enforce_constraints(combined_pop)

        # 2. Calculate objectives
        objs = self.calculate_objectives(combined_pop)

        # 3. Non-Dominated Sort
        fronts = self.fast_non_dominated_sort(objs)

        # 4. Select new population (Front by Front)
        new_pop_indices = []
        capacity = self.POP_SIZE

        for front in fronts:
            if len(new_pop_indices) + len(front) <= capacity:
                new_pop_indices.extend(front)
            else:
                # Last front doesn't fit, select best by Crowding Distance
                needed = capacity - len(new_pop_indices)
                distances = self.crowding_distance_assignment(objs, front)
                # Higher distance (less crowded) is better
                sorted_indices = torch.argsort(distances, descending=True)
                best_in_front = [front[i] for i in sorted_indices[:needed].tolist()]
                new_pop_indices.extend(best_in_front)
                break
        
        new_pop_indices = torch.tensor(new_pop_indices, device=self.device, dtype=torch.long)
        return combined_pop[new_pop_indices]