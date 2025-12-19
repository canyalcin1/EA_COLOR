import torch
import numpy as np
import glob
import os

class GeneticOptimizer:
    def __init__(self, target_lab, input_cols, target_cols, device="cuda"):
        # Cihaz kontrolü (GPU yoksa CPU)
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.target_lab = torch.tensor(target_lab, device=self.device, dtype=torch.float32)
        self.input_cols = input_cols
        self.target_cols = target_cols
        
        # --- MODEL YÜKLEME ---
        self.models = self.load_ensemble_models()
        
        # --- GA AYARLARI (TURBO) ---
        self.POP_SIZE = 150        # 200'den 150'ye çektik (yeterli)
        self.ELITISM_COUNT = 10    # En iyi 10 taneyi koru
        self.MUTATION_RATE = 0.35  
        self.MUTATION_POWER = 0.08 
        
        # Kısıtlamalar
        self.sparsity_limit = 5
        self.allowed_indices = torch.arange(len(input_cols), device=self.device) 

    def load_ensemble_models(self):
        model_files = glob.glob("models/model_fold_*.pt")
        if not model_files: model_files = glob.glob("models_production/model_fold_*.pt")
        
        loaded_models = []
        chk0 = torch.load(model_files[0], map_location=self.device, weights_only=False)
        config = chk0['config']
        
        # Model Sınıfı
        from torch import nn
        class PigmentColorNet(nn.Module):
            def __init__(self, num_pigments, out_dim, cfg):
                super().__init__()
                self.pigment_embedding = nn.Linear(num_pigments, cfg['EMBED_DIM'], bias=False)
                layers = []
                input_size = cfg['EMBED_DIM']
                for _ in range(cfg['N_LAYERS']):
                    layers.append(nn.Linear(input_size, cfg['HIDDEN_DIM']))
                    if cfg['USE_BATCHNORM']: layers.append(nn.BatchNorm1d(cfg['HIDDEN_DIM']))
                    layers.append(nn.SiLU())
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
            # Scaler verilerini GPU tensörlerine çevirelim (Hız için)
            scaler_mean = torch.tensor(chk['scaler'].mean_, device=self.device, dtype=torch.float32)
            scaler_scale = torch.tensor(chk['scaler'].scale_, device=self.device, dtype=torch.float32)
            loaded_models.append((model, scaler_mean, scaler_scale))
            
        return loaded_models

    def set_constraints(self, allowed_indices_list, max_pigments):
        self.allowed_indices = torch.tensor(allowed_indices_list, device=self.device, dtype=torch.long)
        self.sparsity_limit = int(max_pigments)
        # Global maske oluştur (İzin verilmeyenleri sıfırlamak için)
        self.global_mask = torch.zeros(len(self.input_cols), device=self.device, dtype=torch.bool)
        self.global_mask[self.allowed_indices] = True

    def predict_batch(self, population_tensor):
        # 5 Modelin ortalamasını GPU üzerinde al (CPU'ya git-gel yapma)
        total_preds = torch.zeros((population_tensor.shape[0], len(self.target_cols)), device=self.device)
        
        with torch.no_grad():
            for model, s_mean, s_scale in self.models:
                pred_scaled = model(population_tensor)
                # Inverse Transform (Manuel ve GPU'da)
                pred_real = (pred_scaled * s_scale) + s_mean
                total_preds += pred_real
        
        return total_preds / len(self.models)

    def calculate_fitness_batch(self, population):
        preds = self.predict_batch(population)
        
        # Delta E Hesapla
        diff = preds.view(-1, 5, 3) - self.target_lab.view(1, 5, 3)
        delta_e = torch.sqrt(torch.sum(diff**2, dim=2)).mean(dim=1)
        
        # Ceza (Sparsity Penalty) - Vektörel İşlem
        # 0.001'den büyük olanları say
        active_counts = (population > 0.001).sum(dim=1).float()
        # Limit aşımı cezası
        penalty = torch.relu(active_counts - self.sparsity_limit) * 3.5 
        
        total_loss = delta_e + penalty
        return total_loss, delta_e # Loss küçükse iyidir

    def initialize_population(self):
        # Tamamen Tensör üzerinde başlatma
        pop = torch.zeros((self.POP_SIZE, len(self.input_cols)), device=self.device)
        
        # Rastgele pigment seçimi (Maskeleme ile)
        # Her birey için rastgele değerler ata, sonra maske ile istenmeyenleri sil
        rand_vals = torch.rand_like(pop)
        
        # Sadece izin verilen sütunları açık tut
        # İzin verilmeyen sütunları 0 yap
        mask_matrix = self.global_mask.repeat(self.POP_SIZE, 1)
        pop = rand_vals * mask_matrix
        
        # Sparsity sağlamak için rastgele bazılarını sıfırla
        # Her bireyde rastgele %90'ını kapat (ki 4-5 tane kalsın)
        dropout_mask = torch.rand_like(pop) > 0.90
        pop = pop * dropout_mask
        
        # Normalize
        row_sums = pop.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0
        pop = pop / row_sums
        
        return pop

    def evolve_step(self, population):
        """
        TEK ADIMDA TÜM POPÜLASYONU EVRİMLEŞTİR (DÖNGÜSÜZ)
        """
        # 1. Fitness Hesapla
        losses, delta_es = self.calculate_fitness_batch(population)
        
        # 2. Sırala (En iyiden en kötüye)
        sorted_indices = torch.argsort(losses)
        population = population[sorted_indices]
        delta_es = delta_es[sorted_indices]
        
        # En iyi ve en iyi skor
        best_recipe = population[0].clone()
        best_de = delta_es[0].item()
        
        # --- YENİ NESİL OLUŞTURMA (MATRIX OP) ---
        new_pop = torch.zeros_like(population)
        
        # A. Elitism (En iyileri aynen taşı)
        new_pop[:self.ELITISM_COUNT] = population[:self.ELITISM_COUNT]
        
        # B. Crossover (Vektörel)
        # Geri kalan kısım için (POP_SIZE - ELITISM) kadar ebeveyn seç
        num_children = self.POP_SIZE - self.ELITISM_COUNT
        
        # Turnuva seçimi yerine hızlı olsun diye en iyi %50'den rastgele seç
        top_half = population[:self.POP_SIZE // 2]
        
        # Ebeveyn indisleri
        idx1 = torch.randint(0, len(top_half), (num_children,), device=self.device)
        idx2 = torch.randint(0, len(top_half), (num_children,), device=self.device)
        
        parents1 = top_half[idx1]
        parents2 = top_half[idx2]
        
        # Çaprazlama (Arithmetic)
        alpha = torch.rand((num_children, 1), device=self.device)
        children = parents1 * alpha + parents2 * (1 - alpha)
        
        # C. Mutasyon (Vektörel)
        # Tüm çocuklara aynı anda mutasyon maskesi uygula
        mutation_mask = (torch.rand_like(children) < self.MUTATION_RATE)
        noise = torch.randn_like(children) * self.MUTATION_POWER
        
        # Sadece izin verilen sütunlarda gürültü yap
        noise = noise * self.global_mask.view(1, -1)
        
        children = children + (noise * mutation_mask)
        
        # D. Temizlik ve Kısıtlamalar
        children = torch.relu(children) # Negatifleri sil
        children = children * self.global_mask.view(1, -1) # Yasaklıları sil
        
        # Küçük değerleri temizle (Pruning)
        children[children < 0.005] = 0.0
        
        # Normalize
        row_sums = children.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0 # Sıfıra bölmeyi önle
        children = children / row_sums
        
        # Yeni popülasyona ekle
        new_pop[self.ELITISM_COUNT:] = children
        
        return new_pop, best_recipe, best_de