import os

class Config:
    # --- Paths ---
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
    RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'upgraded')
    MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
    
    # --- Environment & Grid ---
    GRID_RESOLUTION_M = 100  # Finer resolution for telecom grade
    ENVIRONMENT_TYPE = 'urban'  # 'urban', 'suburban', 'rural'
    PROPAGATION_MODEL = 'cost231' # 'cost231' or 'fspl'
    
    # --- Network Parameters ---
    TOWER_HEIGHT_M = 30
    USER_HEIGHT_M = 1.5
    FREQUENCY_MHZ = 1800  # 1.8 GHz
    BANDWIDTH_MHZ = 20    # 20 MHz channel
    TRANSMIT_POWER_DBM = 46
    NOISE_FLOOR_DBM = -104 # Standard for 20MHz
    MAX_DISTANCE_KM = 5    # Realistic max range for urban/suburban
    
    # --- Optimization Parameters ---
    POPULATION_SIZE = 40
    NUM_GENERATIONS = 50
    NUM_TOWERS = 15  # Target number of towers
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.8
    TOURNAMENT_SIZE = 3
    
    # --- Genetic Algorithm Weights (for single objective if needed, or Pareto selection) ---
    WEIGHT_COVERAGE = 0.5
    WEIGHT_CAPACITY = 0.3
    WEIGHT_COST = 0.2
    
    # --- Thresholds ---
    RSRP_MIN_DBM = -110  # Minimum usable signal
    SINR_MIN_DB = 0      # Minimum SINR for connectivity
    CELL_CAPACITY_GBPS = 1.0 # 5G Microcell average capacity
    LOAD_PENALTY_WEIGHT = 0.5 # Penalty for overloaded cells
    
    # --- Reproducibility ---
    RANDOM_SEED = 42

    @classmethod
    def setup_directories(cls):
        for path in [cls.PROCESSED_DATA_PATH, cls.RESULTS_PATH, cls.MODELS_PATH]:
            os.makedirs(path, exist_ok=True)
