import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import QUANTILES
from .features import FEATURES


class QuantileGBM:
    """Three gradient-boosted quantile regressors (p10 / p50 / p90), one global model over all horizons."""

    def __init__(self, quantiles=QUANTILES, max_iter=250, learning_rate=0.05, seed=0):
        self.quantiles = quantiles
        self.kw = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=31,
                       min_samples_leaf=100, l2_regularization=1.0, random_state=seed)
        self.models = []

    def fit(self, frame: pd.DataFrame):
        d = frame[frame["y"].notna()]
        if len(d) < 500:
            raise ValueError(f"not enough training rows ({len(d)})")
        self.models = [HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.kw)
                       .fit(d[FEATURES], d["y"]) for q in self.quantiles]
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        P = np.column_stack([m.predict(frame[FEATURES]) for m in self.models])
        P = np.sort(P, axis=1)  # enforce non-crossing quantiles
        return pd.DataFrame(P, columns=[f"p{int(q * 100)}" for q in self.quantiles], index=frame.index)
