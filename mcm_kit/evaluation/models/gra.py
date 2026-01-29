from .base import BaseModel
from typing import Any
import pandas as pd


class GRAModel(BaseModel):
    """灰色关联分析：看样本与理想样本的相似程度，适合小样本"""

    def __init__(self, df: pd.DataFrame, rho: float = 0.5):
        super().__init__(df)
        self.rho = rho  # 分辨系数，通常取0.5

    def fit(self) -> pd.Series:
        # 归一化
        z: pd.DataFrame = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        # 参考序列（取每一列的最大值，即理想样本）
        ref_seq = z.max(axis=0)

        # 计算绝对差矩阵
        diff = abs(z - ref_seq)
        a_min = diff.min().min()
        a_max = diff.max().max()

        # 计算关联系数
        coeff: pd.DataFrame = (a_min + self.rho * a_max) / (diff + self.rho * a_max)
        # 计算关联度（得分）
        return coeff.mean(axis=1)
