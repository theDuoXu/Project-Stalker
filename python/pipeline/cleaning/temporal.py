import pandas as pd
import numpy as np
from ..interfaces import Cleaner
from ..metrics import CLEANING_VIOLATIONS_TOTAL

class TemporalCoherenceCleaner(Cleaner):
    def __init__(self, sensor_type):
        self.sensor_type = sensor_type.upper()

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sensor_type != 'TEMPERATURA':
            return df
            
        # Ensure sorted by time
        df = df.sort_index()
        
        if 'value' not in df.columns:
            return df

        # Calculate Delta T
        # Assuming df has 'value' and index is timestamp
        diff = df['value'].diff().abs()
        
        # Requirement: Delta T > 10ºC regarding "Previous Immediate Record".
        violation_mask = diff > 10.0
        
        count = violation_mask.sum()
        if count > 0:
             CLEANING_VIOLATIONS_TOTAL.labels(violation_type='temporal_coherence').inc(count)
             df.loc[violation_mask, 'value'] = np.nan
             
        return df
