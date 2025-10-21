import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy.spatial.distance import pdist
import geopandas as gpd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- Part 1: Configuration & Setup ---
print("--- Starting Phase 3: AI Optimization Engine (Final Optimized Version) ---")

class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
    RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'phase_3')
    
    POPULATION_SIZE = 50
    NUM_GENERATIONS = 50
    NUM_TOWERS = 15
    MUTATION_RATE = 0.15
    CROSSOVER_RATE = 0.8
    TOURNAMENT_SIZE = 5
    
    TOWER_POWER_DBM = 46; FREQUENCY_MHZ = 1800; TOWER_HEIGHT_M = 30
    GRID_RESOLUTION_M = 200; MIN_TOWER_DISTANCE_CELLS = 10
    RSRP_THRESHOLDS_DBM = {'excellent': -95, 'good': -105, 'fair': -115}
    MAX_LOS_RANGE_KM = 10 

def setup_directories():
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    print("Optimization paths set up.")

def load_feature_layers():
    print("Loading feature layers from Phase 2...")
    try:
        data_files = ['feature_demand_score.tif', 'feature_cost_surface.tif', 'feature_elevation.tif', 'feature_clutter.tif']
        rasters = [rasterio.open(os.path.join(Config.PROCESSED_DATA_PATH, f)) for f in data_files]
        district_boundary = gpd.read_file(os.path.join(Config.PROCESSED_DATA_PATH, 'Boundaries', 'nagpur_boundary.gpkg'))
        data = {'demand': rasters[0].read(1), 'cost': rasters[1].read(1), 'elevation': rasters[2].read(1), 'clutter': rasters[3].read(1),
                'transform': rasters[0].transform, 'crs': rasters[0].crs, 'district_boundary': district_boundary}
        print("Feature layers loaded successfully.")
        return data
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}. Please ensure Phase 2 ran successfully.")
        return None

# --- Part 2: Upgraded & Optimized Coverage Model ---
def precompute_grids(height, width):
    """
    --- THIS IS THE FIX ---
    Returns a dictionary instead of a tuple for clear access by key.
    """
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    return {'y_coords': y_coords, 'x_coords': x_coords}

def calculate_coverage_LoS_optimized(tower_y, tower_x, data, precomputed):
    # Now this line will work correctly
    y_coords, x_coords = precomputed['y_coords'], precomputed['x_coords']
    
    distance_km_grid = np.sqrt((x_coords - tower_x)**2 + (y_coords - tower_y)**2) * Config.GRID_RESOLUTION_M / 1000.0
    distance_km_grid[distance_km_grid == 0] = 0.01

    path_loss = 20 * np.log10(distance_km_grid) + 20 * np.log10(Config.FREQUENCY_MHZ) + 32.45
    is_los = np.ones_like(distance_km_grid, dtype=bool)
    relevant_pixels = distance_km_grid < Config.MAX_LOS_RANGE_KM
    tower_elev = data['elevation'][tower_y, tower_x] + Config.TOWER_HEIGHT_M
    relevant_y, relevant_x = np.where(relevant_pixels)
    
    for y, x in zip(relevant_y, relevant_x):
        if y == tower_y and x == tower_x: continue
        line_len = int(np.hypot(x - tower_x, y - tower_y))
        if line_len < 2: continue
        line_x = np.linspace(tower_x, x, line_len, dtype=int)
        line_y = np.linspace(tower_y, y, line_len, dtype=int)
        terrain_profile = data['elevation'][line_y, line_x]
        line_of_sight_elev = np.linspace(tower_elev, data['elevation'][y, x], line_len)
        if np.any(terrain_profile[1:-1] > line_of_sight_elev[1:-1]):
            is_los[y, x] = False

    path_loss[~is_los] += 20
    path_loss += data['clutter'] * 5
    rsrp = Config.TOWER_POWER_DBM - path_loss
    return rsrp

# --- Part 3: Genetic Algorithm as a Class (with caching) ---
class GeneticAlgorithm:
    def __init__(self, data):
        self.data = data
        self.height, self.width = data['demand'].shape
        self.precomputed_grids = precompute_grids(self.height, self.width)
        self.population = self._initialize_population()
        self.fitness_cache = {}

    def _initialize_population(self):
        print("Creating initial population using demand-based seeding...")
        population = []
        flat_demand = self.data['demand'].flatten()
        valid_indices = np.where(flat_demand > 0.01)[0]
        valid_probs = flat_demand[valid_indices] / np.sum(flat_demand[valid_indices])
        for _ in range(Config.POPULATION_SIZE):
            chosen_indices = np.random.choice(valid_indices, size=Config.NUM_TOWERS, p=valid_probs, replace=False)
            population.append(sorted([np.unravel_index(idx, (self.height, self.width)) for idx in chosen_indices]))
        return population

    def _calculate_fitness(self, individual):
        individual_key = tuple(sorted(individual))
        if individual_key in self.fitness_cache:
            return self.fitness_cache[individual_key]

        all_rsrp = [calculate_coverage_LoS_optimized(y, x, self.data, self.precomputed_grids) for y, x in individual]
        combined_rsrp = np.maximum.reduce(all_rsrp)
        
        covered_area = combined_rsrp >= Config.RSRP_THRESHOLDS_DBM['fair']
        demand_score = np.sum(self.data['demand'][covered_area]) / np.sum(self.data['demand'])
        coverage_score = np.sum(covered_area) / self.data['demand'].size
        cost_score = np.sum([self.data['cost'][y, x] for y, x in individual]) / Config.NUM_TOWERS
        interference_penalty = np.sum(pdist(individual) < Config.MIN_TOWER_DISTANCE_CELLS) * 0.05
        
        fitness = (0.5 * demand_score) + (0.4 * coverage_score) - (0.1 * cost_score) - interference_penalty
        self.fitness_cache[individual_key] = fitness
        return fitness

    def _selection(self, fitnesses):
        tournament = random.sample(list(zip(self.population, fitnesses)), Config.TOURNAMENT_SIZE)
        return max(tournament, key=lambda i: i[1])[0]

    def _crossover(self, p1, p2):
        if random.random() < Config.CROSSOVER_RATE:
            point = random.randint(1, Config.NUM_TOWERS - 1)
            return sorted(p1[:point] + p2[point:]), sorted(p2[:point] + p1[point:])
        return p1, p2

    def _mutate(self, individual):
        mutated_individual = [
            (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if random.random() < Config.MUTATION_RATE
            else gene
            for gene in individual
        ]
        return sorted(mutated_individual)

    def run(self):
        print("Starting Optimized Genetic Algorithm evolution...")
        best_solution, best_fitness = None, -np.inf
        
        for _ in tqdm(range(Config.NUM_GENERATIONS), desc="Evolving Solutions"):
            fitnesses = [self._calculate_fitness(ind) for ind in self.population]
            
            current_best_idx = np.argmax(fitnesses)
            if fitnesses[current_best_idx] > best_fitness:
                best_fitness = fitnesses[current_best_idx]
                best_solution = self.population[current_best_idx]
            
            next_population = [best_solution]  # Elitism
            while len(next_population) < Config.POPULATION_SIZE:
                p1, p2 = self._selection(fitnesses), self._selection(fitnesses)
                c1, c2 = self._crossover(p1, p2)
                next_population.append(self._mutate(c1))
                if len(next_population) < Config.POPULATION_SIZE:
                    next_population.append(self._mutate(c2))
            self.population = next_population
            
        print(f"\nGenetic Algorithm finished. Best fitness: {best_fitness:.4f}")
        return best_solution

# --- Part 4: Visualization ---
def visualize_results(solution, data):
    print("Visualizing the optimal tower placement...")
    final_rsrp_layers = [
        calculate_coverage_LoS_optimized(y, x, data, precompute_grids(data['demand'].shape[0], data['demand'].shape[1]))
        for y, x in solution
    ]
    final_coverage_map = np.maximum.reduce(final_rsrp_layers)
    
    fig, ax = plt.subplots(figsize=(18, 18))
    levels = sorted(list(Config.RSRP_THRESHOLDS_DBM.values())) + [np.max(final_coverage_map)]
    colors = ['#d7191c', '#fdae61', '#abdda4']
    extent = [
        data['transform'].c, data['transform'].c + data['transform'].a * data['demand'].shape[1],
        data['transform'].f + data['transform'].e * data['demand'].shape[0], data['transform'].f
    ]
    
    ax.contourf(final_coverage_map, levels=levels, colors=colors, alpha=0.6, extent=extent)
    data['district_boundary'].to_crs(data['crs']).plot(ax=ax, facecolor='none', edgecolor='white', linewidth=2.5, linestyle='--')
    tower_coords = [(data['transform'] * (x + 0.5, y + 0.5)) for y, x in solution] # Center the tower in the pixel
    tower_xs, tower_ys = zip(*tower_coords)
    ax.scatter(tower_xs, tower_ys, c='cyan', marker='^', s=250, edgecolor='black', zorder=5)

    ax.set_title(f'Optimal 5G/LTE Cell Tower Placement Plan ({len(solution)} Towers)', fontsize=22)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    
    legend_elements = [
        Line2D([0], [0], color='w', marker='^', markerfacecolor='cyan', markeredgecolor='k', markersize=15, label='Optimal Tower Locations'),
        Line2D([0], [0], color='white', lw=2, linestyle='--', label='Nagpur District Boundary'),
        Patch(facecolor=colors[2], alpha=0.6, label=f'Excellent Signal (>{levels[2]} dBm)'),
        Patch(facecolor=colors[1], alpha=0.6, label=f'Good Signal (>{levels[1]} dBm)'),
        Patch(facecolor=colors[0], alpha=0.6, label=f'Fair Signal (>{levels[0]} dBm)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    plt.savefig(os.path.join(Config.RESULTS_PATH, 'optimal_tower_placement_final.png'))
    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    setup_directories()
    feature_data = load_feature_layers()
    if feature_data:
        ga = GeneticAlgorithm(feature_data)
        best_solution_found = ga.run()
        if best_solution_found:
            visualize_results(best_solution_found, feature_data)
    print("\n--- Phase 3: AI Optimization Engine FINISHED ---")