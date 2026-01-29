import pytest
import pandas as pd
import numpy as np
from mcm_kit.evaluation.preprocessing import Preprocessor
from mcm_kit.evaluation.weighting import Weighting
from mcm_kit.evaluation.models.topsis import TopsisModel
from mcm_kit.evaluation.models.gra import GRAModel
from mcm_kit.evaluation.models.coupling import CouplingCoordinationModel
from mcm_kit.evaluation import quick_evaluate


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    模拟5个城市的3个指标：
    - GDP (极大型)
    - 污染指数 (极小型)
    - 适宜温度 (中间型，假设25度最好)
    """
    data = {
        "GDP": [100, 80, 120, 90, 110],
        "Pollution": [30, 50, 20, 40, 10],
        "Temp": [25, 10, 35, 20, 28],
    }
    return pd.DataFrame(data, index=["CityA", "CityB", "CityC", "CityD", "CityE"])


def test_preprocessing(sample_data):
    # 1. 测试正向化：将极小型变为极大型
    pos_pollution = Preprocessor.to_positive(sample_data["Pollution"], type="min")
    assert (
        pos_pollution["CityE"] > pos_pollution["CityB"]
    )  # 污染最小的CityE得分应该最高

    # 2. 测试正向化：中间型（越接近25越好）
    pos_temp = Preprocessor.to_positive(
        sample_data["Temp"], type="mid", extra_params=[25]
    )
    assert pos_temp["CityA"] == 1.0  # 正好25度的CityA得分应该是满分

    # 3. 测试标准化
    norm_df = Preprocessor.normalize(sample_data)
    assert norm_df.max().max() <= 1.0
    assert norm_df.min().min() >= 0.0


def test_weighting(sample_data):
    # GDP和Pollution由于是极大型和极小化的混合，先做个基础归一化再算权重
    df_ready = Preprocessor.normalize(sample_data)

    # 1. 熵权法
    w_entropy = Weighting.entropy_weight(df_ready)
    assert len(w_entropy) == 3
    assert np.isclose(w_entropy.sum(), 1.0)

    # 2. CRITIC权重法
    w_critic = Weighting.critic_weight(df_ready)
    assert len(w_critic) == 3
    assert np.isclose(w_critic.sum(), 1.0)

    # 3. AHP权重法
    # 构造一个3阶一致性矩阵 (1表示同样重要，3表示稍微重要)
    ahp_matrix = np.array([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]])
    w_ahp = Weighting.ahp_weight(ahp_matrix)
    assert w_ahp[0] > w_ahp[1] > w_ahp[2]  # GDP > Pollution > Temp


def test_topsis_model(sample_data):
    # 数据预处理：Pollution取反，Temp转中间型
    df = sample_data.copy()
    df["Pollution"] = Preprocessor.to_positive(df["Pollution"], type="min")
    df["Temp"] = Preprocessor.to_positive(df["Temp"], type="mid", extra_params=[25])

    model = TopsisModel(df)
    scores = model.fit()

    assert isinstance(scores, pd.Series)
    assert scores.idxmax() == "CityE" or scores.idxmax() == "CityC"  # 逻辑上这俩比较强


def test_gra_model(sample_data):
    # 灰色关联分析
    df_norm = Preprocessor.normalize(sample_data)
    model = GRAModel(df_norm)
    scores = model.fit()

    assert scores.min() > 0
    assert scores.max() <= 1.0


def test_coupling_model():
    """测试耦合协调度：假设有两个系统（经济得分、环境得分）"""
    data = {"Economy": [0.9, 0.1, 0.5], "Environment": [0.8, 0.2, 0.5]}
    df = pd.DataFrame(data)
    model = CouplingCoordinationModel(df)
    d_scores = model.fit()

    # 第一行经济环境双高，协调度应最高
    # 第二行双低，协调度应最低
    assert d_scores[0] > d_scores[2] > d_scores[1]


def test_quick_evaluate(sample_data):
    # 最简单的调用方式
    result = quick_evaluate(sample_data, method="topsis")

    assert "Score" in result.columns
    assert "Rank" in result.columns
    assert result["Rank"].iloc[0] == 1  # 排名第一的Rank应该是1
