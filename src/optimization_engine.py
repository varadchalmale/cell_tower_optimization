import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm 

print("--- Starting Phase 3: AI Optimization Engine ---")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_processed_path = os.path.join(project_root, 'data', 'processed')
results_path = os.path.join(project_root, 'results')

print("Loading feature layers from Phase 2...")
try:
    demand_raster = rasterio.open(os.path.join(data_processed_path, 'demand_score.tif'))
    dem_raster = rasterio.open(os.path.join(data_processed_path, 'aligned_dem.tif'))
except FileNotFoundError as e:
    print(f"FATAL ERROR: A required feature file was not found. Please re-run Phase 2 successfully. Details: {e}")
    exit()


demand_grid = demand_raster.read(1)
dem_grid = dem_raster.read(1)
grid_height, grid_width = demand_grid.shape


def create_cost_surface(dem_grid):
    """Creates a cost surface based on elevation and slope."""
    print("Creating cost surface...")
   
    slope_y, slope_x = np.gradient(dem_grid)
    slope = np.sqrt(slope_x**2 + slope_y**2)
    
    
    norm_dem = (dem_grid - np.min(dem_grid)) / (np.max(dem_grid) - np.min(dem_grid) + 1e-9)
    norm_slope = (slope - np.min(slope)) / (np.max(slope) - np.min(slope) + 1e-9)
    
    
    cost_surface = (0.3 * norm_dem) + (0.7 * norm_slope)
    return cost_surface

cost_grid = create_cost_surface(dem_grid)


def hata_path_loss(freq_mhz, ht_m, d_km):
    """Calculates path loss using the Hata model for urban areas."""
    
    hm = 1.5 
    
    ahm = (1.1 * np.log10(freq_mhz) - 0.7) * hm - (1.56 * np.log10(freq_mhz) - 0.8)
    
    
    pl_db = 69.55 + 26.16 * np.log10(freq_mhz) - 13.82 * np.log10(ht_m) - ahm + (44.9 - 6.55 * np.log10(ht_m)) * np.log10(d_km)
    return pl_db

def calculate_coverage(tower_y, tower_x, params):
    """Calculates the received signal strength (RSRP) grid for a single tower."""
   
    y_coords, x_coords = np.mgrid[0:grid_height, 0:grid_width]
    
    
    distance_km = np.sqrt((x_coords - tower_x)**2 + (y_coords - tower_y)**2) * params['grid_resolution_m'] / 1000.0
    distance_km[distance_km == 0] = 0.01

    
    tower_height_above_ground = params['tower_height_m']
    effective_tower_height = tower_height_above_ground + dem_grid[tower_y, tower_x]
    
    path_loss = hata_path_loss(params['frequency_mhz'], effective_tower_height, distance_km)
    
    rsrp = params['tower_power_dbm'] - path_loss
    
    return rsrp


GA_PARAMS = {
    'population_size': 50,    # Number of candidate solutions (chromosomes) in each generation
    'num_generations': 100,   # Number of generations to run the evolution
    'num_towers': 10,         # Number of towers in each candidate solution
    'mutation_rate': 0.1,     # Probability of a tower location being randomly changed
    'crossover_rate': 0.8,    # Probability of two parents creating offspring
    'tournament_size': 5      # Number of individuals to select for tournament selection
}


RF_PARAMS = {
    'tower_power_dbm': 46,       # Standard power for a macro cell tower
    'frequency_mhz': 1800,       # Common LTE band
    'tower_height_m': 30,        # Standard tower height above ground
    'rsrp_threshold_dbm': -110,  # Minimum signal strength for acceptable coverage
    'grid_resolution_m': 200     # Must match the grid from Phase 2
}

def calculate_fitness(chromosome, demand_grid, cost_grid):
    """Calculates the fitness score of a single chromosome (a set of tower locations)."""
    
    all_towers_rsrp = []
    total_cost = 0
    
    for tower_y, tower_x in chromosome:
        rsrp_grid = calculate_coverage(tower_y, tower_x, RF_PARAMS)
        all_towers_rsrp.append(rsrp_grid)
        total_cost += cost_grid[tower_y, tower_x]
    
    combined_rsrp = np.maximum.reduce(all_towers_rsrp)
   
    covered_area = np.sum(combined_rsrp >= RF_PARAMS['rsrp_threshold_dbm'])
    coverage_score = covered_area / (grid_height * grid_width)
    
    
    demand_in_covered_area = np.sum(demand_grid[combined_rsrp >= RF_PARAMS['rsrp_threshold_dbm']])
    total_demand = np.sum(demand_grid)
    demand_score = demand_in_covered_area / total_demand
    
    
    cost_score = total_cost / len(chromosome)
    
    fitness = (0.5 * demand_score) + (0.3 * coverage_score) - (0.2 * cost_score)
    
    return fitness


def create_individual():
    """Creates a single chromosome (a random set of tower locations)."""
    return [(random.randint(0, grid_height-1), random.randint(0, grid_width-1)) for _ in range(GA_PARAMS['num_towers'])]

def selection(population, fitnesses):
    """Selects a parent using tournament selection."""
    tournament = random.sample(list(zip(population, fitnesses)), GA_PARAMS['tournament_size'])
    
    return max(tournament, key=lambda i: i[1])[0]

def crossover(parent1, parent2):
    """Performs one-point crossover between two parents."""
    if random.random() < GA_PARAMS['crossover_rate']:
        point = random.randint(1, GA_PARAMS['num_towers'] - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

def mutate(individual):
    """Performs mutation on an individual."""
    return [(random.randint(0, grid_height-1), random.randint(0, grid_width-1)) if random.random() < GA_PARAMS['mutation_rate'] else gene for gene in individual]

print("Starting Genetic Algorithm...")

population = [create_individual() for _ in range(GA_PARAMS['population_size'])]
best_solution = None
best_fitness = -np.inf


for gen in tqdm(range(GA_PARAMS['num_generations']), desc="Evolving Solutions"):

    fitnesses = [calculate_fitness(ind, demand_grid, cost_grid) for ind in population]
    
    
    current_best_fitness = max(fitnesses)
    if current_best_fitness > best_fitness:
        best_fitness = current_best_fitness
        best_solution = population[fitnesses.index(current_best_fitness)]
    
    
    next_population = [best_solution] 
    
    while len(next_population) < GA_PARAMS['population_size']:
        parent1 = selection(population, fitnesses)
        parent2 = selection(population, fitnesses)
        child1, child2 = crossover(parent1, parent2)
        next_population.append(mutate(child1))
        if len(next_population) < GA_PARAMS['population_size']:
            next_population.append(mutate(child2))
            
    population = next_population

print("Genetic Algorithm finished.")
print(f"Best solution found with fitness score: {best_fitness:.4f}")


print("Visualizing the optimal tower placement...")

final_rsrp_layers = [calculate_coverage(y, x, RF_PARAMS) for y, x in best_solution]
final_coverage_map = np.maximum.reduce(final_rsrp_layers)
covered_pixels = final_coverage_map >= RF_PARAMS['rsrp_threshold_dbm']


fig, ax = plt.subplots(figsize=(15, 15))


im = ax.imshow(demand_grid, cmap='inferno', interpolation='nearest')
fig.colorbar(im, ax=ax, shrink=0.5, label='Demand Score')


coverage_display = np.zeros((*covered_pixels.shape, 4)) 
coverage_display[covered_pixels] = [0, 1, 0, 0.3] 
ax.imshow(coverage_display, interpolation='nearest')

tower_ys, tower_xs = zip(*best_solution)
ax.scatter(tower_xs, tower_ys, c='cyan', marker='^', s=150, edgecolor='black', label=f'Optimal Tower Locations ({GA_PARAMS["num_towers"]})')

ax.set_title('Optimal Cell Tower Placement Plan for Nagpur')
ax.set_xlabel('Grid Cells (West to East)')
ax.set_ylabel('Grid Cells (North to South)')
ax.legend()
plt.savefig(os.path.join(results_path, 'optimal_tower_placement.png'))
plt.show()

print("\n--- Phase 3: AI Optimization Engine Complete ---")