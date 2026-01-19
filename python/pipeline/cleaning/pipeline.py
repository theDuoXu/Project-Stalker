import pandas as pd
from ..interfaces import Cleaner

class CleanerPipeline(Cleaner):
    def __init__(self, cleaners: list[Cleaner]):
        self.cleaners = cleaners
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        for cleaner in self.cleaners:
            df = cleaner.clean(df)
        return df
