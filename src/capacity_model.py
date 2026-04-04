import numpy as np
from src.config import Config

class CapacityModel:
    @staticmethod
    def dbm_to_linear(dbm):
        return 10**((dbm - 30) / 10)

    @staticmethod
    def linear_to_db(linear):
        return 10 * np.log10(linear)

    @classmethod
    def calculate_sinr(cls, serving_rsrp_dbm, interference_rsrp_dbm_list, noise_floor_dbm=Config.NOISE_FLOOR_DBM):
        """
        Calculates SINR in dB.
        interference_rsrp_dbm_list: list of RSRP values from other towers.
        """
        s_linear = cls.dbm_to_linear(serving_rsrp_dbm)
        i_linear = sum([cls.dbm_to_linear(rsrp) for rsrp in interference_rsrp_dbm_list])
        n_linear = cls.dbm_to_linear(noise_floor_dbm)
        
        sinr_linear = s_linear / (i_linear + n_linear + 1e-12)
        return cls.linear_to_db(sinr_linear)

    @staticmethod
    def shannon_capacity(sinr_db, bandwidth_mhz=None):
        """
        Capacity in Mbps.

        Limitation 3 Fix: defaults to TOTAL_BANDWIDTH_MHZ (carrier aggregation
        across all bands) instead of the single-band BANDWIDTH_MHZ.

        Limitation 4 Fix: caps result at MAX_HARDWARE_CAPACITY_GBPS so that
        per-pixel throughput never exceeds what real tower hardware can deliver.
        """
        if bandwidth_mhz is None:
            bandwidth_mhz = Config.TOTAL_BANDWIDTH_MHZ
        sinr_linear = 10 ** (sinr_db / 10)
        raw_mbps = bandwidth_mhz * np.log2(1 + sinr_linear)
        # Hardware cap: convert Gbps ceiling to Mbps for comparison
        hw_cap_mbps = Config.MAX_HARDWARE_CAPACITY_GBPS * 1000
        return np.minimum(raw_mbps, hw_cap_mbps)

    @classmethod
    def apply_backhaul_cap(cls, per_cell_capacity_mbps: dict) -> dict:
        """
        Limitation 4 Fix: apply per-tower backhaul limit.

        per_cell_capacity_mbps: {cell_id -> total_capacity_mbps}
        Returns the same dict with values capped at MAX_BACKHAUL_CAPACITY_GBPS.
        """
        cap_mbps = Config.MAX_BACKHAUL_CAPACITY_GBPS * 1000
        return {cell_id: min(cap_mbps, val) for cell_id, val in per_cell_capacity_mbps.items()}

    @classmethod
    def calculate_cell_load(cls, user_demand_map, serving_indices, capacity_map):
        """
        Estimate load per cell.
        user_demand_map: traffic demand per pixel.
        serving_indices: which cell serves which pixel.
        capacity_map: capacity per pixel.
        """
        unique_cells = np.unique(serving_indices)
        cell_loads = {}
        for cell_id in unique_cells:
            if cell_id < 0: continue # No coverage
            mask = (serving_indices == cell_id)
            total_demand = np.sum(user_demand_map[mask])
            total_capacity = np.mean(capacity_map[mask]) * np.sum(mask) # Simplified total capacity
            # More realistic: sum of capacities per pixel divided by demand? 
            # Actually, capacity per pixel is what a single user gets.
            # Cell total capacity is BW * log2(1 + average SINR) * some factor.
            # Let's say cell_load = sum(demand) / cell_total_capacity
            cell_loads[cell_id] = total_demand / (total_capacity + 1e-6)
        return cell_loads
