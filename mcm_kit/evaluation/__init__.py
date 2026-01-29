from . import models
from . import weighting
from . import preprocessing
from .models import BaseModel, TopsisModel, GRAModel
from pandas import DataFrame


def quick_evaluate(df, method="topsis") -> DataFrame:
    """一键评价接口"""
    if method == "topsis":
        model = TopsisModel(df)
    elif method == "gra":
        model = GRAModel(df)
    else:
        model = BaseModel(df)

    score = model.fit()
    result = score.to_frame(name="Score").sort_values("Score", ascending=False)
    result["Rank"] = range(1, len(result) + 1)
    return result


__version__ = "0.0.2"
__all__ = ["preprocessing", "weighting", "models", "quick_evaluate"]
