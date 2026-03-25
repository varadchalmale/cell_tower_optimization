import numpy as np
import random
from tqdm import tqdm
from scipy.spatial.distance import pdist, cdist
from src.config import Config
from src.propagation_models import PropagationModel
from src.capacity_model import CapacityModel
from src.candidate_site_generator import CandidateSiteGenerator

class MultiObjectiveGA:
    def __init__(self, data, candidates):
        self.data = data
        self.candidates = candidates
        self.height, self.width = data['demand'].shape
        self.population = self._initialize_population()
        self.history = {'fitness': [], 'pareto': []}
        
        # Precompute distance matrices for all candidates to all pixels (Optimization)
        print("Precomputing distance tensors for candidate sites...")
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        self.candidate_rsrp_cache = []
        for y, x in tqdm(self.candidates, desc="Precomputing Propagation"):
            dist_km = np.sqrt((xx - x)**2 + (yy - y)**2) * Config.GRID_RESOLUTION_M / 1000.0
            rsrp = PropagationModel.calculate_rsrp(dist_km)
            self.candidate_rsrp_cache.append(rsrp)
        self.candidate_rsrp_cache = np.array(self.candidate_rsrp_cache)

    def _initialize_population(self):
        population = []
        for _ in range(Config.POPULATION_SIZE):
            # Pick from candidate indices
            indices = np.random.choice(len(self.candidates), Config.NUM_TOWERS, replace=False)
            population.append(indices)
        return np.array(population, dtype=int)

    def _evaluate_individual(self, individual_indices):
        """
        Returns a tuple of (coverage, capacity, -cost) -> all to maximize
        individual_indices: list of indices into self.candidates
        """
        # 1. Coverage & SINR
        # Use precomputed RSRP layers
        all_rsrp = self.candidate_rsrp_cache[individual_indices]
        
        max_rsrp = np.maximum.reduce(all_rsrp)
        covered_mask = max_rsrp >= Config.RSRP_MIN_DBM
        
        # 2. Coverage Objective
        coverage_obj = np.sum(self.data['demand'][covered_mask]) / (np.sum(self.data['demand']) + 1e-6)
        
        # 3. Capacity (SINR based)
        # For each pixel, serving tower is one with max RSRP
        serving_tower_idx_local = np.argmax(all_rsrp, axis=0)
        
        # Interference Calculation
        total_rsrp_linear = np.sum(CapacityModel.dbm_to_linear(all_rsrp), axis=0)
        serving_rsrp_linear = CapacityModel.dbm_to_linear(max_rsrp)
        interference_linear = total_rsrp_linear - serving_rsrp_linear
        
        noise_linear = CapacityModel.dbm_to_linear(Config.NOISE_FLOOR_DBM)
        sinr_linear = serving_rsrp_linear / (interference_linear + noise_linear + 1e-12)
        
        # Shannon Capacity (bits/s/Hz) * Bandwidth
        capacity_map = Config.BANDWIDTH_MHZ * np.log2(1 + sinr_linear)
        
        # Load Analysis & Penalization
        total_capacity_gbps = np.sum(capacity_map[covered_mask]) / 1000.0 # Aggregated
        
        # Cell Load Penalty
        cell_loads = CapacityModel.calculate_cell_load(self.data['demand'], serving_tower_idx_local, capacity_map)
        load_penalty = 0
        for cell_idx, load in cell_loads.items():
            if load > 1.2: # Overloaded
                load_penalty += (load - 1.2) * Config.LOAD_PENALTY_WEIGHT
        
        capacity_obj = (total_capacity_gbps / (Config.NUM_TOWERS * 0.5)) - load_penalty
        
        # 4. Cost Objective
        individual_coords = [self.candidates[i] for i in individual_indices]
        cost_obj = np.sum([self.data['cost'][y, x] for y, x in individual_coords]) / Config.NUM_TOWERS
            
        return (coverage_obj, float(capacity_obj), -float(cost_obj))

    def _is_dominated(self, fit1, fit2):
        """ Returns True if fit1 is dominated by fit2 """
        return all(f2 >= f1 for f1, f2 in zip(fit1, fit2)) and any(f2 > f1 for f1, f2 in zip(fit1, fit2))

    def _fast_non_dominated_sort(self, fitnesses):
        """
        Standard NSGA-II non-dominated sorting.
        Returns a list of fronts (indices).
        """
        size = len(fitnesses)
        S = [[] for _ in range(size)]
        n = [0 for _ in range(size)]
        fronts = [[]]

        for p in range(size):
            for q in range(size):
                if self._is_dominated(fitnesses[p], fitnesses[q]):
                    n[p] += 1
                elif self._is_dominated(fitnesses[q], fitnesses[p]):
                    S[p].append(q)
            
            if n[p] == 0:
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]

    def _crowding_distance(self, fitnesses, front_indices):
        """
        Calculates crowding distance for a front.
        """
        distances = np.zeros(len(front_indices))
        num_objs = len(fitnesses[0])
        
        for m in range(num_objs):
            # Sort front based on objective m
            obj_values = [fitnesses[idx][m] for idx in front_indices]
            sorted_indices = np.argsort(obj_values)
            
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            obj_min = min(obj_values)
            obj_max = max(obj_values)
            norm = obj_max - obj_min + 1e-12
            
            for i in range(1, len(front_indices) - 1):
                distances[sorted_indices[i]] += (obj_values[sorted_indices[i+1]] - obj_values[sorted_indices[i-1]]) / norm
                
        return distances

    def _mutate(self, individual):
        new_ind = list(individual)
        for i in range(len(new_ind)):
            if random.random() < Config.MUTATION_RATE:
                # Snap to a random candidate index
                new_ind[i] = random.randint(0, len(self.candidates)-1)
        return new_ind

    def run(self):
        print(f"Starting NSGA-II Optimization with {len(self.candidates)} candidates...")
        for gen in tqdm(range(Config.NUM_GENERATIONS), desc="NSGA-II Evolving"):
            # 1. Evaluate
            fitnesses = [self._evaluate_individual(ind) for ind in self.population]
            
            # 2. Sort and Crowding
            fronts = self._fast_non_dominated_sort(fitnesses)
            
            # 3. Build next generation
            next_population = []
            for front in fronts:
                if len(next_population) + len(front) <= Config.POPULATION_SIZE:
                    next_population.extend([self.population[idx] for idx in front])
                else:
                    # Fill remaining using crowding distance
                    cd = self._crowding_distance(fitnesses, front)
                    sorted_front = [front[i] for i in np.argsort(cd)[::-1]]
                    next_population.extend([self.population[idx] for idx in sorted_front[:Config.POPULATION_SIZE - len(next_population)]])
                    break
            
            # 4. Offspring generation (Crossover + Mutation)
            offspring = []
            while len(offspring) < Config.POPULATION_SIZE:
                # Tournament selection based on rank (front index)
                idx1, idx2 = random.sample(range(len(next_population)), 2)
                parent = next_population[idx1] # Simplified selection
                child = self._mutate(parent)
                offspring.append(child)
            
            self.population = np.array(offspring, dtype=int)
            self.history['pareto'].append(fronts[0]) # Store first front indices
            
        # Final result from the last generation
        final_fitnesses = [self._evaluate_individual(ind) for ind in self.population]
        final_fronts = self._fast_non_dominated_sort(final_fitnesses)
        best_indices = final_fronts[0]
        
        return [self.population[i] for i in best_indices], [final_fitnesses[i] for i in best_indices]
