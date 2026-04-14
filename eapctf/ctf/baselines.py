from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold


@dataclass
class RollingLinearBaseline:
    """Shared rolling/expanding linear-model baseline for CTF-style panels."""

    model_family: str
    window_length: int = 120
    train_scheme: str = "rolling"
    min_train_rows: int | None = None
    rank_transform: bool = True
    id_col: str = "id"
    eom_col: str = "eom"
    eom_ret_col: str = "eom_ret"
    target_col: str = "ret_exc_lead1m"
    test_flag_col: str = "ctff_test"
    weight_col: str = "w"
    selected_params_: dict[str, dict[str, float]] = field(default_factory=dict, init=False, repr=False)
    last_train_row_count_: int | None = field(default=None, init=False, repr=False)

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
        self.selected_params_.clear()
        self.last_train_row_count_ = None

        for d in pf_dates:
            train = self._slice_training_window(chars, d, features)
            if len(train) < min_rows:
                continue
            self.last_train_row_count_ = len(train)

            x_train = train[features].to_numpy(dtype=float)
            y_train = train[self.target_col].to_numpy(dtype=float)
            model, params = self.fit_estimator(x_train, y_train)
            self.selected_params_[str(pd.Timestamp(d).date())] = params

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

    def _slice_training_window(self, chars: pd.DataFrame, d, features: list[str]) -> pd.DataFrame:
        if self.train_scheme not in {"rolling", "expanding"}:
            raise ValueError(f"unsupported train_scheme: {self.train_scheme}")
        if self.train_scheme == "rolling":
            window_start = d + pd.DateOffset(days=1) - pd.DateOffset(months=self.window_length) - pd.DateOffset(days=1)
            train_mask = (chars[self.eom_ret_col] < d) & (chars[self.eom_ret_col] >= window_start)
        else:
            train_mask = chars[self.eom_ret_col] < d
        return chars.loc[train_mask, [self.id_col, self.eom_col, self.target_col, *features]].copy()

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
    """Minimal rolling/expanding OLS point baseline for CTF-style panels."""

    model_family: str = "ols"

    def fit_estimator(self, x_train: np.ndarray, y_train: np.ndarray):
        beta = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        return beta, {"alpha": 0.0}


@dataclass
class _RegularizedLinearBaseline(RollingLinearBaseline):
    alpha: float = 1.0
    alpha_grid: list[float] | None = None
    cv_folds: int = 0

    def fit_estimator(self, x_train: np.ndarray, y_train: np.ndarray):
        if self.cv_folds and self.cv_folds >= 2 and self.alpha_grid:
            params = self._select_params_via_cv(x_train, y_train)
        else:
            params = self.default_params()
        model = self.build_model(**params)
        model.fit(x_train, y_train)
        return model, {k: float(v) for k, v in params.items()}

    def _select_params_via_cv(self, x_train: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
        n_splits = min(self.cv_folds, len(y_train))
        if n_splits < 2:
            return self.default_params()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        best_params = None
        best_score = float("inf")
        for params in self.param_grid():
            fold_scores: list[float] = []
            for train_idx, val_idx in kf.split(x_train):
                model = self.build_model(**params)
                model.fit(x_train[train_idx], y_train[train_idx])
                pred = np.asarray(model.predict(x_train[val_idx]), dtype=float)
                mse = float(np.mean((y_train[val_idx] - pred) ** 2))
                fold_scores.append(mse)
            score = float(np.mean(fold_scores))
            if score < best_score:
                best_score = score
                best_params = params
        return best_params or self.default_params()

    def default_params(self) -> dict[str, float]:
        return {"alpha": self.alpha}

    def param_grid(self) -> list[dict[str, float]]:
        assert self.alpha_grid is not None
        return [{"alpha": alpha} for alpha in self.alpha_grid]

    def build_model(self, **params):
        raise NotImplementedError


@dataclass
class RollingRidgeBaseline(_RegularizedLinearBaseline):
    model_family: str = "ridge"
    alpha: float = 1.0

    def build_model(self, **params):
        return Ridge(alpha=params["alpha"], fit_intercept=False, random_state=None)


@dataclass
class RollingLassoBaseline(_RegularizedLinearBaseline):
    model_family: str = "lasso"
    alpha: float = 0.001
    max_iter: int = 10000

    def build_model(self, **params):
        return Lasso(alpha=params["alpha"], fit_intercept=False, max_iter=self.max_iter, random_state=42)


@dataclass
class RollingElasticNetBaseline(_RegularizedLinearBaseline):
    model_family: str = "elastic_net"
    alpha: float = 0.001
    l1_ratio: float = 0.5
    l1_ratio_grid: list[float] | None = None
    max_iter: int = 10000

    def default_params(self) -> dict[str, float]:
        return {"alpha": self.alpha, "l1_ratio": self.l1_ratio}

    def param_grid(self) -> list[dict[str, float]]:
        alpha_grid = self.alpha_grid or [self.alpha]
        l1_grid = self.l1_ratio_grid or [self.l1_ratio]
        return [
            {"alpha": alpha, "l1_ratio": l1_ratio}
            for alpha, l1_ratio in product(alpha_grid, l1_grid)
        ]

    def build_model(self, **params):
        return ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            fit_intercept=False,
            max_iter=self.max_iter,
            random_state=42,
        )
