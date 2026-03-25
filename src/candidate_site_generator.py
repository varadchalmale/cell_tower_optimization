import numpy as np
from src.config import Config

class CandidateSiteGenerator:
    @staticmethod
    def generate_sites(demand_score, cost_surface, elevation, num_candidates=200, min_sep_cells=5):
        """
        Creates a list of feasible tower locations.
        Focuses on high demand, low cost, and reasonable elevation.
        Enforces a minimum separation between candidate sites.
        """
        # Score pixels for candidacy (weighted approach)
        candidacy_score = (demand_score * 0.7) - (cost_surface * 0.3)
        
        # Flatten and get top indices as a starting pool
        flat_scores = candidacy_score.flatten()
        top_indices = np.argsort(flat_scores)[-num_candidates*5:] # Larger pool to filter from
        
        height, width = demand_score.shape
        potential_candidates = [np.unravel_index(idx, (height, width)) for idx in top_indices[::-1]]
        
        final_candidates = []
        for cand in potential_candidates:
            if len(final_candidates) >= num_candidates:
                break
            
            # Check minimum separation from already selected candidates
            if not final_candidates:
                final_candidates.append(cand)
                continue
                
            y, x = cand
            dists = [np.sqrt((y - cy)**2 + (x - cx)**2) for cy, cx in final_candidates]
            if min(dists) >= min_sep_cells:
                final_candidates.append(cand)
        
        print(f"Generated {len(final_candidates)} candidate sites with min separation {min_sep_cells} cells.")
        return final_candidates

    @staticmethod
    def snap_to_candidate(coords, candidates):
        """
        Snaps a given (y, x) to the nearest candidate site.
        """
        y, x = coords
        dists = [np.sqrt((y - cy)**2 + (x - cx)**2) for cy, cx in candidates]
        return candidates[np.argmin(dists)]
