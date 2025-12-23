import torch
import numpy as np
from optimizer_engine import EvolutionaryOptimizer

class NSGA2Optimizer(EvolutionaryOptimizer):
    def __init__(self, target_lab, input_cols, target_cols, device="cpu"):
        super().__init__(target_lab, input_cols, target_cols, device)
        # NSGA-II Ayarları
        self.POP_SIZE = 200 # Pareto için biraz daha kalabalık olmalı
        
    def calculate_objectives(self, population):
        """
        NSGA-II için 2 Hedef Döndürür:
        1. Delta E (Minimize)
        2. Pigment Sayısı (Minimize)
        """
        # Gradient istemiyoruz, sadece sıralama yapacağız
        with torch.no_grad():
            preds = self.predict_batch(population)
            diff = preds.view(-1, 5, 3) - self.target_lab.view(1, 5, 3)
            
            # Hedef 1: Delta E
            delta_e = torch.sqrt(torch.sum(diff**2, dim=2)).mean(dim=1)
            
            # Hedef 2: Active Pigment Count (Soft Count)
            # 0.001'den büyük olanları say
            # Türevsiz işlem olduğu için direkt sayabiliriz
            active_counts = (population > 0.001).sum(dim=1).float()
            
            # İki hedefi birleştir: (N, 2)
            objectives = torch.stack((delta_e, active_counts), dim=1)
            
        return objectives

    def fast_non_dominated_sort(self, objectives):
        """
        Popülasyonu 'Front' (Cephe) katmanlarına ayırır.
        Front 0: En iyiler (Kimse tarafından domine edilmeyenler)
        """
        N = objectives.shape[0]
        dominates = (objectives.unsqueeze(1) < objectives.unsqueeze(0)).all(dim=2) 
        # a, b'yi domine eder mi? (a < b her hedefte)
        
        # Basitleştirilmiş sıralama (PyTorch ile hızlandırılmış)
        ranks = torch.zeros(N, dtype=torch.long, device=self.device)
        current_front = []
        
        # Her birey için kaç kişi onu domine ediyor?
        domination_counts = (objectives.unsqueeze(0) > objectives.unsqueeze(1)).all(dim=2).sum(dim=1)
        
        # Count'u 0 olanlar ilk cephedir
        current_front = torch.where(domination_counts == 0)[0].tolist()
        
        fronts = [current_front]
        visited = torch.zeros(N, dtype=torch.bool, device=self.device)
        visited[current_front] = True
        
        # Diğer cepheleri bul (Basit mantık: Ön cepheyi sil, kalanların en iyisini al)
        while True:
            # Henüz atanmamış olanlar
            remaining_indices = torch.where(~visited)[0]
            if len(remaining_indices) == 0:
                break
                
            # Kalanlar arasında Pareto hesabı
            sub_objs = objectives[remaining_indices]
            # Kalanlar içinde domine edilme sayısı
            sub_dom_counts = (sub_objs.unsqueeze(0) > sub_objs.unsqueeze(1)).all(dim=2).sum(dim=1)
            
            next_front_local = torch.where(sub_dom_counts == 0)[0]
            next_front_global = remaining_indices[next_front_local]
            
            fronts.append(next_front_global.tolist())
            visited[next_front_global] = True
            
        return fronts

    def crowding_distance_assignment(self, objectives, front_indices):
        """
        Aynı cephedeki bireylerin çeşitliliğini korumak için mesafe hesabı.
        """
        l = len(front_indices)
        if l == 0: return torch.tensor([], device=self.device)
        
        distances = torch.zeros(l, device=self.device)
        front_objs = objectives[front_indices]
        
        # Her hedef için ayrı ayrı sırala ve mesafe ölç
        for m in range(2): # 2 Hedefimiz var
            sorted_idx = torch.argsort(front_objs[:, m])
            sorted_objs = front_objs[sorted_idx, m]
            
            # Uç noktalar sonsuz uzaklıkta (Asla silinmesin diye)
            distances[sorted_idx[0]] = 1e9
            distances[sorted_idx[-1]] = 1e9
            
            # Aradakilerin mesafesi: (Sonraki - Önceki) / (Max - Min)
            diff = (sorted_objs[2:] - sorted_objs[:-2])
            scale = sorted_objs[-1] - sorted_objs[0]
            if scale == 0: scale = 1.0
            
            distances[sorted_idx[1:-1]] += diff / scale
            
        return distances

    def evolve_nsga2(self, population):
        # 1. Yavru Üret (Crossover & Mutation) - Klasik GA gibi
        offspring_pop, _, _ = self.step_ga(population) 
        # step_ga elitism yapıyor ama biz tüm havuzu istiyoruz, o yüzden biraz hileli:
        # NSGA-II'de ebeveyn + yavru havuzu birleşir (R + Q)
        
        combined_pop = torch.cat((population, offspring_pop), dim=0)
        # Constraintleri uygula
        combined_pop = self.enforce_constraints(combined_pop)
        
        # 2. Hedefleri Hesapla
        objs = self.calculate_objectives(combined_pop)
        
        # 3. Non-Dominated Sort
        fronts = self.fast_non_dominated_sort(objs)
        
        # 4. Yeni Popülasyonu Seç (Front by Front)
        new_pop_indices = []
        capacity = self.POP_SIZE
        
        for front in fronts:
            if len(new_pop_indices) + len(front) <= capacity:
                new_pop_indices.extend(front)
            else:
                # Son cephe sığmıyor, Crowding Distance ile en iyileri seç
                needed = capacity - len(new_pop_indices)
                distances = self.crowding_distance_assignment(objs, front)
                # Mesafesi büyük olan (kalabalık olmayan) iyidir
                sorted_indices = torch.argsort(distances, descending=True)
                best_in_front = [front[i] for i in sorted_indices[:needed].tolist()]
                new_pop_indices.extend(best_in_front)
                break
        
        new_pop_indices = torch.tensor(new_pop_indices, device=self.device, dtype=torch.long)
        return combined_pop[new_pop_indices]