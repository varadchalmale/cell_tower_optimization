"""
RF propagation models: COST-231 Hata and 3GPP UMa.

The key export is `vectorized_rsrp_matrix()` — a shared, environment-aware,
fading-margin-corrected RSRP computation used by nsga2.py, multi_tier.py,
and validate.py.  All three modules call this single function to guarantee
consistency.
"""

import numpy as np


# ======================================================================
# Shared vectorised RSRP computation (canonical formula)
# ======================================================================

def vectorized_rsrp_matrix(grid_df, towers_df, config):
    """
    Compute RSRP from every tower to every grid point.

    Uses COST-231 Hata with **environment-dependent** corrections:
      - urban:    cm = 0 (medium-sized city; cm=3 reserved for megacities)
      - suburban: urban path loss minus suburban correction
      - rural:    urban path loss minus rural correction

    Includes a configurable **shadow fading margin** (default 8 dB)
    subtracted from RSRP for 95 % location probability.

    Args:
        grid_df:    DataFrame with columns 'x', 'y', 'urban_class'
        towers_df:  DataFrame with columns 'x', 'y'
        config:     dict (parsed config.yaml)

    Returns:
        rsrp_matrix: np.ndarray, shape (n_grid, n_towers), dtype float32
    """
    fc = config["rf_params"]["frequency_mhz"]
    tx = config["rf_params"]["transmit_power_dbm"]
    ht = config["rf_params"]["antenna_height_m"]
    hr = config["rf_params"]["receiver_height_m"]
    gain = config["rf_params"].get("antenna_gain_dbi", 18)
    fading = config["optimization"].get("shadow_fading_margin_db", 8)

    # COST-231 Hata constants (cm = 0, urban medium city)
    ahr = 3.2 * (np.log10(11.75 * hr)) ** 2 - 4.97
    base_A = 46.3 + 33.9 * np.log10(fc) - 13.82 * np.log10(ht) - ahr
    B = 44.9 - 6.55 * np.log10(ht)

    # Environment correction offsets (applied to base_A per grid point)
    suburban_offset = -(2 * (np.log10(fc / 28)) ** 2 + 5.4)
    rural_offset = -(4.78 * (np.log10(fc)) ** 2 - 18.33 * np.log10(fc) + 40.94)

    # Per-grid-point environment correction vector
    env = grid_df["urban_class"].values if "urban_class" in grid_df.columns else np.full(len(grid_df), "urban")
    env_corr = np.zeros(len(grid_df), dtype=np.float32)
    env_corr[env == "suburban"] = suburban_offset
    env_corr[env == "rural"] = rural_offset
    # urban stays 0

    # Coordinates
    gx = grid_df["x"].values[:, np.newaxis].astype(np.float64)
    gy = grid_df["y"].values[:, np.newaxis].astype(np.float64)
    tx_arr = towers_df["x"].values[np.newaxis, :].astype(np.float64)
    ty_arr = towers_df["y"].values[np.newaxis, :].astype(np.float64)

    d_km = np.sqrt((gx - tx_arr) ** 2 + (gy - ty_arr) ** 2) / 1000.0
    d_km = np.maximum(d_km, 0.02)   # 20 m minimum (COST-231 validity floor)
    np.clip(d_km, None, 20.0, out=d_km)  # 20 km max (model validity ceiling)

    # Path loss: each grid point gets its own A constant via env_corr
    # PL[i,j] = (base_A + env_corr[i]) + B * log10(d_km[i,j])
    A_vec = (base_A + env_corr)[:, np.newaxis]  # (n_grid, 1)
    pl = A_vec + B * np.log10(d_km)
    np.maximum(pl, 38.0, out=pl)

    rsrp = (tx + gain - pl - fading).astype(np.float32)
    return rsrp


def compute_coverage_radius_km(config, environment="urban"):
    """Compute theoretical coverage radius for a given environment."""
    fc = config["rf_params"]["frequency_mhz"]
    tx = config["rf_params"]["transmit_power_dbm"]
    ht = config["rf_params"]["antenna_height_m"]
    hr = config["rf_params"]["receiver_height_m"]
    gain = config["rf_params"].get("antenna_gain_dbi", 18)
    fading = config["optimization"].get("shadow_fading_margin_db", 8)
    threshold = config["optimization"].get("coverage_rsrp_threshold_dbm", -110)

    ahr = 3.2 * (np.log10(11.75 * hr)) ** 2 - 4.97
    base_A = 46.3 + 33.9 * np.log10(fc) - 13.82 * np.log10(ht) - ahr
    B = 44.9 - 6.55 * np.log10(ht)

    if environment == "suburban":
        offset = -(2 * (np.log10(fc / 28)) ** 2 + 5.4)
    elif environment == "rural":
        offset = -(4.78 * (np.log10(fc)) ** 2 - 18.33 * np.log10(fc) + 40.94)
    else:
        offset = 0.0

    A = base_A + offset
    max_pl = tx + gain - threshold - fading
    # A + B * log10(d) = max_pl  -->  d = 10^((max_pl - A) / B)
    log_d = (max_pl - A) / B
    radius_km = min(20.0, 10 ** log_d)
    return max(0.02, radius_km)


# ======================================================================
# Original per-link propagation model (kept for reference / single-pair)
# ======================================================================

class PropagationModel:
    def __init__(self, config):
        self.config = config
        self.freq_mhz = config["rf_params"]["frequency_mhz"]
        self.tx_power_dbm = config["rf_params"]["transmit_power_dbm"]
        self.ht_m = config["rf_params"]["antenna_height_m"]
        self.hr_m = config["rf_params"]["receiver_height_m"]
        self.model_type = config["rf_params"]["propagation_model"]
        self.thermal_noise_dbm = -104

    def _cost231_hata(self, d_km, environment="urban"):
        if environment in ("urban", "dense_urban"):
            if self.freq_mhz >= 400:
                ahr = 3.2 * (np.log10(11.75 * self.hr_m)) ** 2 - 4.97
            else:
                ahr = 8.29 * (np.log10(1.54 * self.hr_m)) ** 2 - 1.1
            cm = 3 if environment == "dense_urban" else 0
        else:
            ahr = ((1.1 * np.log10(self.freq_mhz) - 0.7) * self.hr_m
                   - (1.56 * np.log10(self.freq_mhz) - 0.8))
            cm = 0

        pl_urban = (46.3 + 33.9 * np.log10(self.freq_mhz)
                    - 13.82 * np.log10(self.ht_m) - ahr
                    + (44.9 - 6.55 * np.log10(self.ht_m)) * np.log10(d_km) + cm)

        if environment == "suburban":
            pl = pl_urban - 2 * (np.log10(self.freq_mhz / 28)) ** 2 - 5.4
        elif environment == "rural":
            pl = (pl_urban - 4.78 * (np.log10(self.freq_mhz)) ** 2
                  + 18.33 * np.log10(self.freq_mhz) - 40.94)
        else:
            pl = pl_urban
        return max(38.0, pl)

    def calculate_sinr(self, rsrp_matrix):
        rsrp_lin = 10 ** (rsrp_matrix / 10.0)
        noise_lin = 10 ** (self.thermal_noise_dbm / 10.0)
        sinr_matrix = np.zeros_like(rsrp_matrix)
        for i in range(rsrp_matrix.shape[0]):
            for j in range(rsrp_matrix.shape[1]):
                signal = rsrp_lin[i, j]
                interference = np.sum(rsrp_lin[i, :]) - signal
                sinr_lin = signal / (interference + noise_lin)
                sinr_matrix[i, j] = 10 * np.log10(sinr_lin + 1e-12)
        return sinr_matrix
