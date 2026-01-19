import pandas as pd
import numpy as np
from ..interfaces import Cleaner
from ..metrics import CLEANING_VIOLATIONS_TOTAL
from .config import HARD_LIMITS

class HardLimitsCleaner(Cleaner):
    def __init__(self, sensor_type):
        self.sensor_type = sensor_type.upper()
        self.limits = HARD_LIMITS.get(self.sensor_type)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.limits:
            return df
        
        min_val, max_val = self.limits
        
        # Identify violations
        if 'value' in df.columns:
            violation_mask = (df['value'] < min_val) | (df['value'] > max_val)
            
            count = violation_mask.sum()
            if count > 0:
                CLEANING_VIOLATIONS_TOTAL.labels(violation_type='hard_limit').inc(count)
                # Force to NaN
                df.loc[violation_mask, 'value'] = np.nan
            
        return df
