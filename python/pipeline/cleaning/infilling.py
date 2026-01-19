import pandas as pd
import numpy as np
from ..interfaces import Cleaner
from ..metrics import VALUES_IMPUTED_TOTAL
from .config import GROUP_A_INERTIAL, GROUP_B_BIOLOGICAL, GROUP_C_EVENT

class SmartInfillingCleaner(Cleaner):
    def __init__(self, sensor_type):
        self.sensor_type = sensor_type.upper()
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Resample to ensure regular intervals
        # Split into numeric and non-numeric to handle metadata preservation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 1. Resample Numeric (Mean)
        df_numeric = df[numeric_cols].resample('1h').mean()
        
        # 2. Resample Metadata & Info (Take first)
        # We need to preserve 'metadata_json', 'metric', 'unit' if they exist
        non_numeric_cols = ['metadata_json', 'metric', 'unit', 'station_id', 'sensor_id']
        existing_non_numeric = [c for c in non_numeric_cols if c in df.columns]
        
        if existing_non_numeric:
            df_info = df[existing_non_numeric].resample('1h').first()
        else:
            df_info = pd.DataFrame(index=df_numeric.index)

        # Recombine
        # Note: df_numeric might be missing columns if empty
        df_resampled = df_numeric
        if not df_info.empty:
            df_resampled = pd.concat([df_numeric, df_info], axis=1)

        method = None
        limit_hours = 0
        order = None
        
        if self.sensor_type in GROUP_A_INERTIAL:
            method = 'linear'
            limit_hours = 6
        elif self.sensor_type in GROUP_B_BIOLOGICAL:
            method = 'spline' # Spline requires order
            order = 3
            limit_hours = 3
        elif self.sensor_type in GROUP_C_EVENT:
            method = 'linear'
            limit_hours = 2
        
        if method and limit_hours > 0:
             # Convert limit hours to number of 1h intervals
            limit_intervals = int(limit_hours)
            
            # Interpolate only numeric columns
            # Count NaNs before
            nans_before = df_resampled['value'].isna().sum() if 'value' in df_resampled.columns else 0

            if method == 'spline':
                 df_filled_numeric = df_resampled[numeric_cols].interpolate(method=method, order=order, limit=limit_intervals, limit_direction='both')
            else:
                 df_filled_numeric = df_resampled[numeric_cols].interpolate(method=method, limit=limit_intervals, limit_direction='both')
            
            # Assign back to preserve structure
            for col in numeric_cols:
                df_resampled[col] = df_filled_numeric[col]
            
            # Count NaNs after to calculate imputed
            nans_after = df_resampled['value'].isna().sum() if 'value' in df_resampled.columns else 0
            imputed_count = nans_before - nans_after
            if imputed_count > 0:
                VALUES_IMPUTED_TOTAL.labels(method=method).inc(imputed_count)
        
        return df_resampled
