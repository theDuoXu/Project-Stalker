import pandas as pd
from ..interfaces import Cleaner

class FlatLineCleaner(Cleaner):
    """
    Detects sensors that 'flat line' (send same value for hours).
    Logic: Derivative = 0 on a large window.
    """
    def __init__(self, sensor_id):
        self.sensor_id = sensor_id

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Implement advanced infilling logic.
        # Detect derivate = 0 windows.
        # Use previous data + other sensors to interpolate.
        return df
