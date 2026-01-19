from abc import ABC, abstractmethod
import pandas as pd

class Cleaner(ABC):
    """
    Abstract Base Class for Data Cleaners.
    Follows the Decorator/Chain of Responsibility pattern where cleaners can be stacked.
    """
    
    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning or infilling logic to the DataFrame.
        Expected DataFrame structure: indexed by timestamp, with 'value' column or similar.
        """
        pass
