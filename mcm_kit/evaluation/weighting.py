import numpy as np
import pandas as pd


class Weighting:
    @staticmethod
    def entropy_weight(df: pd.DataFrame) -> pd.Series:
        z = (df - df.min()) / (df.max() - df.min()) + 1e-9
        p = z / z.sum(axis=0)
        e = -1 / np.log(len(df)) * (p * np.log(p)).sum(axis=0)
        d = 1 - e
        return d / d.sum()

    @staticmethod
    def critic_weight(df: pd.DataFrame) -> pd.Series:
        """CRITIC 权重法：考虑对比强度和指标间的冲突性"""
        # 1. 标准化
        z = (df - df.min()) / (df.max() - df.min())
        # 2. 标准差（对比强度）
        std = z.std(axis=0)
        # 3. 相关系数矩阵（冲突性）
        corr = z.corr()
        # 4. 计算信息量
        f = std * (1 - corr).sum(axis=0)
        return f / f.sum()

    @staticmethod
    def ahp_weight(matrix: np.ndarray) -> np.ndarray:
        eig_val, eig_vec = np.linalg.eig(matrix)
        max_idx = np.argmax(np.real(eig_val))
        weight = np.real(eig_vec[:, max_idx])
        return weight / weight.sum()
