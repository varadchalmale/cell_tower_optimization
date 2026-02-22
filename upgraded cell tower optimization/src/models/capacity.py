import numpy as np

class CapacityModel:
    def __init__(self, config):
        self.config = config
        self.bandwidth_mhz = self.config['rf_params']['bandwidth_mhz']
        # Typical 4G/5G max PRBs and efficiency
        self.max_throughput_mbps_per_hz = 0.5 # Sub-optimal mapping proxy

    def compute_spectral_efficiency(self, sinr_db):
        """Shannon mapping from SINR (dB) to Spectral Efficiency (bps/Hz)"""
        # Truncate SINR based on realistic modem capabilities (e.g., max 30dB, min -10dB)
        sinr_clipped = np.clip(sinr_db, -10, 30)
        sinr_linear = 10**(sinr_clipped / 10.0)
        # Shannon limit: log2(1 + S/N)
        # Empirical discount on Shannon (real-world is ~60% of Shannon capacity)
        se = 0.6 * np.log2(1 + sinr_linear)
        return se

    def convert_to_throughput(self, se_matrix):
        """Convert Spectral Efficiency matrix to Throughput (Mbps)"""
        # Throughput = SE * Bandwidth
        # For a cell, total bandwidth is shared. This gives max theoretical per user if all BW allocated
        throughput_matrix = se_matrix * self.bandwidth_mhz
        return throughput_matrix

    def compute_cell_load(self, demand_array, throughput_matrix, assignment_matrix):
        """Compute the load for each cell based on assigned demand.
           demand_array: Traffic demand per grid point
           throughput_matrix: Max throughput if user had full BW
           assignment_matrix: Rx -> Tx (boolean matrix or probability matrix)
        """
        # Load of a cell = Sum_{users in cell} (demand / max_throughput)
        # This tells us the fraction of time/bandwidth the cell needs to serve its users
        
        num_cells = assignment_matrix.shape[1]
        cell_loads = np.zeros(num_cells)
        
        for c in range(num_cells):
            assigned_users = assignment_matrix[:, c]
            user_demand = demand_array[assigned_users]
            
            # Use max throughput available per user
            user_max_tput = np.maximum(1e-3, throughput_matrix[assigned_users, c])
            
            # The bandwidth fraction required by each user
            req_fractions = user_demand / user_max_tput
            cell_loads[c] = np.sum(req_fractions)
            
        return cell_loads

    def compute_penalty_and_actual_throughput(self, cell_loads, demand_array, assignment_matrix):
        """Return the capacity objective and user satisfaction based on cell loads."""
        # Max load = 1.0 (100% bandwidth utilized)
        # If load > 1.0, the cell is overloaded, throughput drops for everyone
        
        num_cells = len(cell_loads)
        capacity_penalty = 0.0
        
        user_satisfied_demand = np.zeros(len(demand_array))
        
        for c in range(num_cells):
            assigned_users = assignment_matrix[:, c]
            total_demand = demand_array[assigned_users].sum()
            
            if cell_loads[c] <= 1.0:
                # Fully capable of serving users
                user_satisfied_demand[assigned_users] = demand_array[assigned_users]
            else:
                # Overloaded: proportional fair sharing
                # Penalty grows exponentially with overload
                capacity_penalty += (cell_loads[c] - 1.0) ** 2
                
                shrink_factor = 1.0 / cell_loads[c]
                user_satisfied_demand[assigned_users] = demand_array[assigned_users] * shrink_factor
                
        # Objective to maximize: total satisfied demand
        total_satisfied = user_satisfied_demand.sum()
        return total_satisfied, capacity_penalty
