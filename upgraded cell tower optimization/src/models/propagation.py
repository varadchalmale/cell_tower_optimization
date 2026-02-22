import numpy as np
import math

class PropagationModel:
    def __init__(self, config):
        self.config = config
        self.freq_mhz = self.config['rf_params']['frequency_mhz']
        self.tx_power_dbm = self.config['rf_params']['transmit_power_dbm']
        self.ht_m = self.config['rf_params']['antenna_height_m']
        self.hr_m = self.config['rf_params']['receiver_height_m']
        self.model_type = self.config['rf_params']['propagation_model']
        self.thermal_noise_dbm = -104 # typical noise floor for 20MHz bandwidth
        
    def _cost231_hata(self, d_km, environment="urban"):
        """Calculate path loss using COST-231 Hata model."""
        # Calculate a(hR) depending on environment
        if environment in ["urban", "dense_urban"]:
            if self.freq_mhz >= 400:
                ahr = 3.2 * (np.log10(11.75 * self.hr_m))**2 - 4.97
            else:
                ahr = 8.29 * (np.log10(1.54 * self.hr_m))**2 - 1.1
            cm = 3 if environment == "dense_urban" else 0
        else: # suburban or rural
            ahr = (1.1 * np.log10(self.freq_mhz) - 0.7) * self.hr_m - (1.56 * np.log10(self.freq_mhz) - 0.8)
            cm = 0

        # Base path loss equation for urban
        pl_urban = (46.3 + 33.9 * np.log10(self.freq_mhz) 
                    - 13.82 * np.log10(self.ht_m) - ahr 
                    + (44.9 - 6.55 * np.log10(self.ht_m)) * np.log10(d_km) + cm)
                    
        # Apply corrections
        if environment == "suburban":
            pl = pl_urban - 2 * (np.log10(self.freq_mhz / 28))**2 - 5.4
        elif environment == "rural":
            pl = pl_urban - 4.78 * (np.log10(self.freq_mhz))**2 + 18.33 * np.log10(self.freq_mhz) - 40.94
        else:
            pl = pl_urban
            
        return max(38.0, pl) # Avoid pathloss dropping below free space approx at very short dist

    def _3gpp_uma(self, d_m, is_los=True):
        """Calculate path loss using simplified 3GPP TR 38.901 UMa model."""
        fc_ghz = self.freq_mhz / 1000.0
        d_m = max(10, d_m) # min distance 10m
        hE = 1.0 # Effective environment height
        
        # LOS
        pl_los = 28.0 + 22 * np.log10(d_m) + 20 * np.log10(fc_ghz)
        
        if is_los:
            return pl_los
            
        # NLOS
        pl_nlos = 13.54 + 39.08 * np.log10(d_m) + 20 * np.log10(fc_ghz) - 0.6 * (self.ht_m - hE)
        return max(pl_los, pl_nlos)

    def compute_terrain_diffraction(self, elev_tx, elev_rx, d_m):
        """Simplistic terrain diffraction penalty based on elevation difference."""
        # Simple proxy: if Rx is much higher than Tx, negative penalty (gain)
        # If Rx is much lower, positive penalty (loss)
        # Assuming earth curvature proxy and basic slope masking
        elev_diff = elev_tx - elev_rx
        # Rough proxy for diffraction loss: ~0.1 dB per meter of negative elevation diff
        penalty = 0
        if elev_diff < -10:
            penalty = abs(elev_diff + 10) * 0.1
        # Bonus for elevation advantage
        elif elev_diff > 20:
            penalty = - (elev_diff - 20) * 0.05
        return np.clip(penalty, -5, 20)

    def calculate_rsrp(self, tx_x, tx_y, tx_elev, rx_x, rx_y, rx_elev, rx_env="urban"):
        """Calculate Reference Signal Received Power for a single Tx-Rx pair."""
        # Euclidean distance in meters
        d_m = np.sqrt((tx_x - rx_x)**2 + (tx_y - rx_y)**2)
        d_km = max(0.001, d_m / 1000.0)
        
        # Terrain checking proxy: If Rx elevation is significantly higher, assume partial LOS
        # Real LOS requires full profile, here we use elevation diff as heuristic
        los_flag = True if tx_elev + self.ht_m >= rx_elev + self.hr_m - 10 else False
        
        # Terrain diffraction loss
        diff_loss = self.compute_terrain_diffraction(tx_elev + self.ht_m, rx_elev + self.hr_m, d_m)
        
        # Base path loss
        if self.model_type == "COST231":
            pl = self._cost231_hata(d_km, environment=rx_env)
        else:
            pl = self._3gpp_uma(d_m, is_los=los_flag)
            
        # Total propagation loss
        total_loss = pl + diff_loss
        
        # Calculate RSRP = Tx_Power - Path_Loss + Gains - Penetration/Misc
        # Assume 15dBi antenna gain, 0 penetrations (outdoor proxy for RSRP)
        rx_rsrp = self.tx_power_dbm + 15.0 - total_loss
        return rx_rsrp

    def calculate_sinr(self, rsrp_matrix):
        """Calculate SINR for all receivers given an RSRP matrix (shape: Rx x Tx)."""
        # Linear units
        rsrp_lin = 10 ** (rsrp_matrix / 10.0)
        noise_lin = 10 ** (self.thermal_noise_dbm / 10.0)
        
        sinr_matrix = np.zeros_like(rsrp_matrix)
        for i in range(rsrp_matrix.shape[0]):
            for j in range(rsrp_matrix.shape[1]):
                signal = rsrp_lin[i, j]
                # Interference is sum of all other RSRPs
                interference = np.sum(rsrp_lin[i, :]) - signal
                sinr_lin = signal / (interference + noise_lin)
                sinr_matrix[i, j] = 10 * np.log10(sinr_lin + 1e-12)
        return sinr_matrix
