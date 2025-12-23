import torch
import numpy as np
import glob
import copy
from torch import nn

class EvolutionaryOptimizer:
    def __init__(self, target_lab, input_cols, target_cols, device="cpu"):
        self.device = device
        self.target_lab = torch.tensor(target_lab, device=self.device, dtype=torch.float32)
        self.input_cols = input_cols
        self.target_cols = target_cols
        
        # Modelleri yükle
        self.models = self.load_ensemble_models()
        
        # Varsayılan Ayarlar
        self.POP_SIZE = 100
        self.sparsity_limit = 8
        self.penalty_weight = 1.5 # <--- Yeni bir değişken (Ceza katsayısı düşürüldü)
        self.global_mask = torch.ones(len(input_cols), device=self.device, dtype=torch.bool)

    def load_ensemble_models(self):
        model_files = glob.glob("models/model_fold_*.pt")
        if not model_files: model_files = glob.glob("models_production/model_fold_*.pt")
        
        loaded_models = []
        chk0 = torch.load(model_files[0], map_location=self.device, weights_only=False)
        config = chk0['config']
        
        # --- MODEL SINIFI ---
        class PigmentColorNet(nn.Module):
            def __init__(self, num_pigments, out_dim, cfg):
                super().__init__()
                self.pigment_embedding = nn.Linear(num_pigments, cfg['EMBED_DIM'], bias=False)
                layers = []
                input_size = cfg['EMBED_DIM']
                
                for _ in range(cfg['N_LAYERS']):
                    layers.append(nn.Linear(input_size, cfg['HIDDEN_DIM']))
                    if cfg.get('USE_BATCHNORM', False): 
                        layers.append(nn.BatchNorm1d(cfg['HIDDEN_DIM']))
                    layers.append(nn.SiLU())
                    if cfg.get('DROPOUT', 0) > 0:
                        layers.append(nn.Dropout(cfg['DROPOUT']))
                    input_size = cfg['HIDDEN_DIM']
                layers.append(nn.Linear(input_size, out_dim))
                self.net = nn.Sequential(*layers)
                
            def forward(self, x):
                return self.net(self.pigment_embedding(x))

        for m_path in model_files:
            chk = torch.load(m_path, map_location=self.device, weights_only=False)
            model = PigmentColorNet(len(self.input_cols), len(self.target_cols), config).to(self.device)
            model.load_state_dict(chk['model_state'])
            model.eval()
            s_mean = torch.tensor(chk['scaler'].mean_, device=self.device, dtype=torch.float32)
            s_scale = torch.tensor(chk['scaler'].scale_, device=self.device, dtype=torch.float32)
            loaded_models.append((model, s_mean, s_scale))
            
        return loaded_models

    def set_constraints(self, allowed_indices_list, max_pigments):
        self.allowed_indices = torch.tensor(allowed_indices_list, device=self.device, dtype=torch.long)
        self.sparsity_limit = int(max_pigments)
        self.global_mask = torch.zeros(len(self.input_cols), device=self.device, dtype=torch.bool)
        self.global_mask[self.allowed_indices] = True

    def predict_batch(self, population_tensor):
        """
        DİKKAT: Burada 'no_grad' OLMAZ! Çünkü Fine-Tune işlemi gradient istiyor.
        """
        total_preds = torch.zeros((population_tensor.shape[0], len(self.target_cols)), device=self.device)
        
        # Döngü içinde gradient takibi açık kalmalı (Fine-tune için)
        for model, s_mean, s_scale in self.models:
            pred_scaled = model(population_tensor)
            pred_real = (pred_scaled * s_scale) + s_mean
            total_preds += pred_real
            
        return total_preds / len(self.models)

    def calculate_fitness_batch(self, population):
        with torch.no_grad():
            preds = self.predict_batch(population)
            diff = preds.view(-1, 5, 3) - self.target_lab.view(1, 5, 3)
            delta_e = torch.sqrt(torch.sum(diff**2, dim=2)).mean(dim=1)
            
            # CEZA MEKANİZMASI GÜNCELLEMESİ
            # Count > 0.001 olanları say
            active_counts = (population > 0.001).sum(dim=1).float()
            
            # Eğer limit (8) aşılırsa her fazla pigment için sadece 1.5 puan ceza ver (Eskiden 10'du)
            # Bu sayede algoritma "Delta E'yi 5 puan düşürecekse 1 pigment fazla kullanayım" diyebilecek.
            penalty = torch.relu(active_counts - self.sparsity_limit) * self.penalty_weight
        
        return delta_e + penalty, delta_e

    def enforce_constraints(self, pop):
        # Eğer tensor gradient gerektiriyorsa (Fine-tune sırasında), işlem yaparken dikkat etmeliyiz
        # Ancak burada genellikle detach edilmiş veriyle çalışacağız.
        if pop.requires_grad:
            # Gradient akışını bozmadan işlem yapmak zor, o yüzden fine-tune içinde manuel constraint var.
            # Bu fonksiyon sadece GA/DE için.
            pop = pop.detach()

        pop = pop * self.global_mask.view(1, -1)
        pop = torch.relu(pop)
        pop[pop < 0.005] = 0.0
        row_sums = pop.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0
        pop = pop / row_sums
        return pop

    def initialize_population(self):
        pop = torch.rand((self.POP_SIZE, len(self.input_cols)), device=self.device)
        dropout_mask = torch.rand_like(pop) > 0.85
        pop = pop * dropout_mask
        return self.enforce_constraints(pop)

    def step_ga(self, population):
        losses, delta_es = self.calculate_fitness_batch(population)
        sorted_indices = torch.argsort(losses)
        population = population[sorted_indices]
        delta_es = delta_es[sorted_indices]
        
        best_recipe, best_de = population[0].clone(), delta_es[0].item()
        
        elitism_count = 5
        new_pop = torch.zeros_like(population)
        new_pop[:elitism_count] = population[:elitism_count]
        
        num_children = self.POP_SIZE - elitism_count
        parents = population[:self.POP_SIZE // 2]
        
        idx1 = torch.randint(0, len(parents), (num_children,), device=self.device)
        idx2 = torch.randint(0, len(parents), (num_children,), device=self.device)
        
        alpha = torch.rand((num_children, 1), device=self.device)
        children = parents[idx1] * alpha + parents[idx2] * (1 - alpha)
        
        mask = (torch.rand_like(children) < 0.2)
        noise = torch.randn_like(children) * 0.1 
        children += mask * noise
        
        new_pop[elitism_count:] = self.enforce_constraints(children)
        return new_pop, best_recipe, best_de

    def step_de(self, population, F=0.5, CR=0.7):
        indices = torch.argsort(torch.rand(self.POP_SIZE, 3, device=self.device), dim=0)
        a = population[indices[:, 0]]
        b = population[indices[:, 1]]
        c = population[indices[:, 2]]
        
        mutant = a + F * (b - c)
        mutant = self.enforce_constraints(mutant)
        
        cross_points = torch.rand_like(population) < CR
        trial = torch.where(cross_points, mutant, population)
        trial = self.enforce_constraints(trial)
        
        fitness_pop, de_pop = self.calculate_fitness_batch(population)
        fitness_trial, de_trial = self.calculate_fitness_batch(trial)
        
        better_mask = fitness_trial < fitness_pop
        new_pop = torch.where(better_mask.unsqueeze(1), trial, population)
        
        best_idx = fitness_trial.argmin()
        if fitness_trial[best_idx] < fitness_pop.min():
             return new_pop, trial[best_idx], de_trial[best_idx].item()
        else:
             best_idx = fitness_pop.argmin()
             return new_pop, population[best_idx], de_pop[best_idx].item()

    def init_pso(self):
        self.velocities = torch.zeros((self.POP_SIZE, len(self.input_cols)), device=self.device)
        self.pbest = torch.zeros((self.POP_SIZE, len(self.input_cols)), device=self.device)
        self.pbest_scores = torch.ones(self.POP_SIZE, device=self.device) * 9999.0
        self.gbest = None
        self.gbest_score = 9999.0

    def step_pso(self, population, w=0.5, c1=1.5, c2=1.5):
        if not hasattr(self, 'velocities'): self.init_pso()
        
        scores, delta_es = self.calculate_fitness_batch(population)
        
        better_mask = scores < self.pbest_scores
        self.pbest[better_mask] = population[better_mask]
        self.pbest_scores[better_mask] = scores[better_mask]
        
        min_score, min_idx = torch.min(self.pbest_scores, dim=0)
        if min_score < self.gbest_score:
            self.gbest_score = min_score.item()
            self.gbest = self.pbest[min_idx].clone()
            
        r1 = torch.rand_like(population)
        r2 = torch.rand_like(population)
        
        self.velocities = (w * self.velocities + 
                           c1 * r1 * (self.pbest - population) + 
                           c2 * r2 * (self.gbest.unsqueeze(0) - population))
        
        new_pop = population + self.velocities
        new_pop = self.enforce_constraints(new_pop)
        
        return new_pop, self.gbest, self.gbest_score

    # --- MEMETIC STEP: FINE TUNING (İNCE AYAR) ---
    def fine_tune(self, best_recipe, steps=100, lr=0.01):
        """
        Gradient Descent ile son dokunuş.
        """
        # Reçeteyi türev alınabilir hale getir
        # (Clone ve Detach önemli, graph'ı koparıp yeni bir tane başlatıyoruz)
        optimized_recipe = best_recipe.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([optimized_recipe], lr=lr)
        
        # Kullanılmayan pigmentleri baskılamak için maske
        with torch.no_grad():
            zero_mask = (best_recipe <= 0.001)
        
        history = []
        
        for i in range(steps):
            optimizer.zero_grad()
            
            # 1. Tahmin (Artık gradient takibi AÇIK)
            preds = self.predict_batch(optimized_recipe.unsqueeze(0))
            
            # 2. Loss: Delta E
            diff = preds.view(-1, 5, 3) - self.target_lab.view(1, 5, 3)
            loss = torch.sqrt(torch.sum(diff**2, dim=2)).mean()
            
            # 3. Loss: Sparsity Koruması (Sıfır olanları elleme cezası)
            # Eğer model sıfır olması gereken bir pigmenti artırırsa ceza ver
            loss += torch.sum(torch.abs(optimized_recipe) * zero_mask) * 1000.0
            
            loss.backward()
            optimizer.step()
            
            # Constraintleri manuel uygula (Gradienti bozmadan data üzerinde)
            with torch.no_grad():
                # Negatifleri sil
                optimized_recipe.data = torch.relu(optimized_recipe.data)
                # Yasaklıları sıfırla
                optimized_recipe.data[zero_mask] = 0.0
                # Toplamı 1 yap (Renormalizasyon)
                s = optimized_recipe.data.sum()
                if s > 1e-6: optimized_recipe.data /= s
            
            history.append(loss.item())
            
            if loss.item() < 0.1: # Hedefe ulaştık
                break
                
        return optimized_recipe.detach(), loss.item()