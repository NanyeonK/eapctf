from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge


@dataclass
class RollingLinearBaseline:
    """Shared rolling linear-model baseline for CTF-style panels."""

    model_family: str
    window_length: int = 120
    min_train_rows: int | None = None
    rank_transform: bool = True
    id_col: str = "id"
    eom_col: str = "eom"
    eom_ret_col: str = "eom_ret"
    target_col: str = "ret_exc_lead1m"
    test_flag_col: str = "ctff_test"
    weight_col: str = "w"

    def prepare_data(self, chars: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        chars = chars.copy()
        chars[self.eom_col] = pd.to_datetime(chars[self.eom_col])
        chars[self.eom_ret_col] = pd.to_datetime(chars[self.eom_ret_col])

        if not self.rank_transform:
            return chars

        for feat in features:
            is_zero = chars[feat] == 0
            chars[feat] = chars.groupby(self.eom_col)[feat].transform(
                lambda x: x.rank(method="max", pct=True)
            )
            chars.loc[is_zero, feat] = 0
            chars[feat] = chars[feat].fillna(0.5)

        chars[features] = chars[features] - 0.5
        return chars

    def run(self, chars: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        chars = self.prepare_data(chars, features)
        pf_dates = chars.loc[chars[self.test_flag_col], self.eom_ret_col].sort_values().unique()
        results: list[pd.DataFrame] = []
        min_rows = self.min_train_rows if self.min_train_rows is not None else len(features) + 1

        for d in pf_dates:
            window_start = d + pd.DateOffset(days=1) - pd.DateOffset(months=self.window_length) - pd.DateOffset(days=1)
            train_mask = (chars[self.eom_ret_col] < d) & (chars[self.eom_ret_col] >= window_start)
            train = chars.loc[train_mask, [self.id_col, self.eom_col, self.target_col, *features]].copy()
            if len(train) < min_rows:
                continue

            x_train = train[features].to_numpy(dtype=float)
            y_train = train[self.target_col].to_numpy(dtype=float)
            model = self.fit_estimator(x_train, y_train)

            test_slice = chars.loc[chars[self.eom_ret_col] == d, [self.id_col, *features]].copy()
            if test_slice.empty:
                continue
            preds = self.predict_estimator(model, test_slice[features].to_numpy(dtype=float))
            weights = self._construct_weights(preds)
            results.append(
                pd.DataFrame(
                    {
                        self.id_col: test_slice[self.id_col].to_numpy(),
                        self.eom_col: d,
                        self.weight_col: weights,
                    }
                )
            )

        if not results:
            return pd.DataFrame(columns=[self.id_col, self.eom_col, self.weight_col])
        out = pd.concat(results, ignore_index=True)
        out[self.eom_col] = pd.to_datetime(out[self.eom_col])
        return out

    def fit_estimator(self, x_train: np.ndarray, y_train: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def predict_estimator(model, x_test: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict"):
            return np.asarray(model.predict(x_test), dtype=float)
        return np.asarray(x_test @ model, dtype=float)

    @staticmethod
    def _construct_weights(predictions: np.ndarray) -> np.ndarray:
        mu = float(np.mean(predictions))
        sigma = float(np.std(predictions))
        return (predictions - mu) / (sigma + 1e-8)


@dataclass
class RollingOLSBaseline(RollingLinearBaseline):
    """Minimal rolling OLS point baseline for CTF-style panels."""

    model_family: str = "ols"

    def fit_estimator(self, x_train: np.ndarray, y_train: np.ndarray):
        return np.linalg.lstsq(x_train, y_train, rcond=None)[0]


@dataclass
class RollingRidgeBaseline(RollingLinearBaseline):
    model_family: str = "ridge"
    alpha: float = 1.0

    def fit_estimator(self, x_train: np.ndarray, y_train: np.ndarray):
        model = Ridge(alpha=self.alpha, fit_intercept=False, random_state=None)
        model.fit(x_train, y_train)
        return model


@dataclass
class RollingLassoBaseline(RollingLinearBaseline):
    model_family: str = "lasso"
    alpha: float = 0.001
    max_iter: int = 10000

    def fit_estimator(self, x_train: np.ndarray, y_train: np.ndarray):
        model = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter, random_state=42)
        model.fit(x_train, y_train)
        return model


@dataclass
class RollingElasticNetBaseline(RollingLinearBaseline):
    model_family: str = "elastic_net"
    alpha: float = 0.001
    l1_ratio: float = 0.5
    max_iter: int = 10000

    def fit_estimator(self, x_train: np.ndarray, y_train: np.ndarray):
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=False,
            max_iter=self.max_iter,
            random_state=42,
        )
        model.fit(x_train, y_train)
        return model
