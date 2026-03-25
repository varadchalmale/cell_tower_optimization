import numpy as np
from src.config import Config

class PropagationModel:
    @staticmethod
    def cost231_hata(distance_km, frequency_mhz=Config.FREQUENCY_MHZ, 
                    hte=Config.TOWER_HEIGHT_M, hre=Config.USER_HEIGHT_M, 
                    env_type=Config.ENVIRONMENT_TYPE):
        """
        Implements COST-231 Hata propagation model.
        Returns path loss in dB.
        """
        # Ensure distance is at least small value to avoid log(0)
        d = np.maximum(distance_km, 0.01)
        f = frequency_mhz
        
        # a(hre) correction factor
        a_hre = (1.1 * np.log10(f) - 0.7) * hre - (1.56 * np.log10(f) - 0.8)
        
        # Cm correction factor
        if env_type == 'urban':
            cm = 3
        else:
            cm = 0
            
        path_loss = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(hte) - a_hre + \
                    (44.9 - 6.55 * np.log10(hte)) * np.log10(d) + cm
                    
        return path_loss

    @staticmethod
    def fspl(distance_km, frequency_mhz=Config.FREQUENCY_MHZ):
        """
        Free Space Path Loss (fallback).
        """
        d = np.maximum(distance_km, 0.01)
        f = frequency_mhz
        return 20 * np.log10(d) + 20 * np.log10(f) + 32.45

    @classmethod
    def calculate_rsrp(cls, distance_km, tx_power_dbm=Config.TRANSMIT_POWER_DBM, model=None):
        if model is None:
            model = Config.PROPAGATION_MODEL
            
        if model == 'cost231':
            loss = cls.cost231_hata(distance_km)
        else:
            loss = cls.fspl(distance_km)
        return tx_power_dbm - loss
