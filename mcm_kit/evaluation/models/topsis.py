from .base import BaseModel
from typing import Optional
from ..weighting import Weighting
import numpy as np
import pandas as pd


class TopsisModel(BaseModel):
    """经典的TOPSIS评价模型"""

    def __init__(self, df: pd.DataFrame, weights: Optional[pd.Series] = None):
        super().__init__(df)
        self.weights: pd.Series = (
            weights if weights is not None else Weighting.entropy_weight(df)
        )

    def fit(self) -> pd.Series:
        # 1. 向量归一化 (TOPSIS的标准做法)
        norm_df = self.df.div(np.sqrt((self.df**2).sum(axis=0)), axis=1)

        # 2. 计算加权矩阵
        weighted_df = norm_df.multiply(self.weights, axis=1)

        # 3. 确定最优解和最劣解
        ideal_best = weighted_df.max()
        ideal_worst = weighted_df.min()

        # 4. 计算欧氏距离
        d_best = np.sqrt(((weighted_df - ideal_best) ** 2).sum(axis=1))
        d_worst = np.sqrt(((weighted_df - ideal_worst) ** 2).sum(axis=1))

        # 5. 计算综合评价指标 (C值)
        score = d_worst / (d_best + d_worst)
        return pd.Series(score, index=self.df.index)
