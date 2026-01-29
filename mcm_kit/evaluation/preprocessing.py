import numpy as np
import pandas as pd
from typing import Optional, List


class Preprocessor:
    """数据预处理：正向化与标准化"""

    @staticmethod
    def to_positive(
        data: pd.Series, type: str = "min", extra_params: Optional[List[float]] = None
    ) -> pd.Series:
        """
        正向化处理
        extra_params:
            - mid型: [best_val]
            - range型: [low, high]
        """
        if type == "min":
            return data.max() - data
        elif type == "mid":
            best_val = extra_params[0] if extra_params else data.mean()
            return 1 - abs(data - best_val) / abs(data - best_val).max()
        elif type == "range":
            low, high = extra_params if extra_params else (data.min(), data.max())

            def range_logic(x):
                if x < low:
                    return 1 - (low - x) / (low - data.min())
                elif x > high:
                    return 1 - (x - high) / (data.max() - high)
                else:
                    return 1.0

            return data.apply(range_logic)
        return data

    @staticmethod
    def normalize(df: pd.DataFrame, method: str = "min-max") -> pd.DataFrame:
        if method == "min-max":
            return (df - df.min()) / (df.max() - df.min())
        elif method == "z-score":
            return (df - df.mean()) / df.std()
        return df
