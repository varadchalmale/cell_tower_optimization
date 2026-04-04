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

    # Limitation 3 Fix: Multi-band model (700 / 1800 / 2600 MHz)
    # Each band has its own frequency, bandwidth, and TX power.
    # The optimizer picks the best-RSRP band per pixel (coverage layer = 700 MHz range,
    # capacity layer = carrier aggregation across all bands).
    BANDS = [
        {'name': '700MHz',  'freq_mhz': 700,  'bw_mhz': 10, 'tx_power_dbm': 46},
        {'name': '1800MHz', 'freq_mhz': 1800, 'bw_mhz': 20, 'tx_power_dbm': 46},
        {'name': '2600MHz', 'freq_mhz': 2600, 'bw_mhz': 20, 'tx_power_dbm': 43},
    ]
    TOTAL_BANDWIDTH_MHZ = 50  # 10 + 20 + 20 — used for Shannon capacity (carrier aggregation)

    # Legacy single-band parameters kept for backward compatibility
    FREQUENCY_MHZ = 1800
    BANDWIDTH_MHZ = 20

    TRANSMIT_POWER_DBM = 46
    NOISE_FLOOR_DBM = -104  # Standard for 20 MHz
    MAX_DISTANCE_KM = 5     # Realistic max range for urban/suburban

    # Limitation 2 Fix: Indoor coverage modelling
    # ITU-R P.2109 / 3GPP TR 38.901 recommend 15–25 dB penetration loss for
    # concrete buildings.  18 dB is the standard mid-range value.
    INDOOR_PENETRATION_LOSS_DB = 18   # applied where clutter (building) density > 0.5

    # Limitation 4 Fix: Tower capacity constraints
    # Real macro towers are limited by hardware MIMO capacity and backhaul link.
    MAX_HARDWARE_CAPACITY_GBPS = 2.0  # per-tower radio hardware ceiling
    MAX_BACKHAUL_CAPACITY_GBPS = 1.0  # per-tower backhaul (fibre/microwave) limit
    OVERLOAD_PENALTY_WEIGHT    = 0.3  # subtracted from capacity objective per overloaded tower

    # Limitation 6 Fix: Temporal demand profiles
    # Demand multipliers relative to peak (1.0).  Design scenario selects which
    # multiplier is applied before optimisation — 'peak' ensures the network
    # handles worst-case load.
    TEMPORAL_PROFILES = {
        'peak':    1.00,   # 08:00–10:00 and 17:00–20:00 weekday
        'daytime': 0.60,   # 10:00–17:00 business hours
        'evening': 0.75,   # 20:00–23:00
        'night':   0.15,   # 23:00–06:00
    }
    DESIGN_SCENARIO = 'peak'  # optimise for worst-case traffic
    
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
