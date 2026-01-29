from .base import BaseModel
import pandas as pd
import numpy as np


class CouplingCoordinationModel(BaseModel):
    """耦合协调度模型：分析多个系统间的交互作用"""

    def fit(self) -> pd.Series:
        # 假设 df 的每一列已经是各个子系统的综合得分（0-1之间）
        # 1. 计算耦合度 C
        n = self.df.shape[1]
        product = self.df.prod(axis=1)
        sum_val = self.df.sum(axis=1) / n
        # 修正计算公式，防止数值溢出
        c = np.power(product / np.power(sum_val, n), 1 / n)

        # 2. 计算综合评价指数 T (默认子系统权重平等)
        t = self.df.mean(axis=1)

        # 3. 计算耦合协调度 D
        d = np.sqrt(c * t)
        return d
