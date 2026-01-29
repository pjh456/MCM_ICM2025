from pandas import DataFrame, Series


class BaseModel:
    def __init__(self, df: DataFrame):
        self.df = df

    def fit(self) -> Series:
        """所有模型必须实现fit方法并返回Series"""
        raise NotImplementedError
