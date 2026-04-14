"""Uncertainty method module.

Active rebuild contract:
- point prediction and uncertainty must be emitted together
- post-hoc two-stage `q_hat` attachment is rejected as the primary design
- benchmark path should call one unified model interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UncertaintyConfig:
    """Configuration for joint point + uncertainty prediction models."""

    target: str = "joint_point_uncertainty"
    representation: str = "stock_level_uncertainty"
    fail_closed: bool = True


@dataclass(frozen=True)
class PredictionWithUncertainty:
    """Unified prediction object emitted at predict time."""

    point: np.ndarray
    uncertainty: np.ndarray

    def __post_init__(self) -> None:
        if self.point.shape != self.uncertainty.shape:
            raise ValueError("point and uncertainty must share shape")


class JointUncertaintyModel(ABC):
    """Unified model interface for CTF models with uncertainty output.

    Required lifecycle:
    - fit(...)
    - predict_with_uncertainty(...)

    The package rebuild rejects architectures where uncertainty is learned only
    as a detached post-processing step after a point-only model has already been
    finalized.
    """

    @abstractmethod
    def fit(self, X: object, y: object) -> "JointUncertaintyModel":
        """Fit a model that will later emit both point and uncertainty."""

    @abstractmethod
    def predict_with_uncertainty(self, X: object) -> PredictionWithUncertainty:
        """Return point prediction and stock-level uncertainty together."""


@dataclass
class ArchivedIPCAResidualVolatilityModel(JointUncertaintyModel):
    """First concrete unified benchmark model for the rebuild.

    Point output is the archived IPCA benchmark weight on each `(id, eom)` row.
    Uncertainty output is trailing stock-level daily return volatility computed
    only from data available up to the portfolio formation month-end.

    This keeps the benchmark anchor fixed while forcing the rebuild to emit
    point and uncertainty together from one fitted object.
    """

    lookback_months: int = 12
    uncertainty_floor: float = 1e-6
    id_col: str = "id"
    eom_col: str = "eom"
    w_col: str = "w"
    date_col: str = "date"
    ret_col: str = "ret_exc"
    _point_lookup: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _uncertainty_lookup: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def fit(self, X: object, y: object) -> "ArchivedIPCAResidualVolatilityModel":
        weights = self._coerce_frame(X, required=[self.id_col, self.eom_col, self.w_col], name="weights")
        daily = self._coerce_frame(y, required=[self.id_col, self.date_col, self.ret_col], name="daily_ret")

        weights = weights[[self.id_col, self.eom_col, self.w_col]].copy()
        weights[self.eom_col] = pd.to_datetime(weights[self.eom_col])
        weights = weights.sort_values([self.eom_col, self.id_col]).reset_index(drop=True)

        daily = daily[[self.id_col, self.date_col, self.ret_col]].copy()
        daily[self.date_col] = pd.to_datetime(daily[self.date_col])
        daily = daily.sort_values([self.date_col, self.id_col]).reset_index(drop=True)

        self._point_lookup = weights
        self._uncertainty_lookup = self._build_uncertainty_lookup(weights, daily)
        return self

    def predict_with_uncertainty(self, X: object) -> PredictionWithUncertainty:
        request = self._coerce_frame(X, required=[self.id_col, self.eom_col], name="request")
        if self._point_lookup is None or self._uncertainty_lookup is None:
            raise RuntimeError("model must be fit before prediction")

        request = request[[self.id_col, self.eom_col]].copy()
        request[self.eom_col] = pd.to_datetime(request[self.eom_col])

        merged = request.merge(
            self._point_lookup,
            on=[self.id_col, self.eom_col],
            how="left",
            validate="one_to_one",
        ).merge(
            self._uncertainty_lookup,
            on=[self.id_col, self.eom_col],
            how="left",
            validate="one_to_one",
        )

        if merged[self.w_col].isna().any():
            missing = merged.loc[merged[self.w_col].isna(), [self.id_col, self.eom_col]]
            raise ValueError(f"missing point weights for request rows: {missing.to_dict(orient='records')[:5]}")

        merged["uncertainty"] = merged["uncertainty"].fillna(self.uncertainty_floor).clip(lower=self.uncertainty_floor)
        return PredictionWithUncertainty(
            point=merged[self.w_col].to_numpy(dtype=float),
            uncertainty=merged["uncertainty"].to_numpy(dtype=float),
        )

    def predict_frame(self, X: object) -> pd.DataFrame:
        request = self._coerce_frame(X, required=[self.id_col, self.eom_col], name="request")
        request = request[[self.id_col, self.eom_col]].copy()
        request[self.eom_col] = pd.to_datetime(request[self.eom_col])
        prediction = self.predict_with_uncertainty(request)
        out = request.copy()
        out[self.w_col] = prediction.point
        out["uncertainty"] = prediction.uncertainty
        return out

    def _build_uncertainty_lookup(self, weights: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
        records: list[pd.DataFrame] = []
        unique_eoms = weights[self.eom_col].drop_duplicates().sort_values()

        for eom in unique_eoms:
            window_start = eom - pd.DateOffset(months=self.lookback_months)
            window = daily[(daily[self.date_col] <= eom) & (daily[self.date_col] > window_start)]
            if window.empty:
                continue

            vol = (
                window.groupby(self.id_col)[self.ret_col]
                .std(ddof=1)
                .rename("uncertainty")
                .reset_index()
            )
            vol["uncertainty"] = vol["uncertainty"].fillna(self.uncertainty_floor).clip(lower=self.uncertainty_floor)
            vol[self.eom_col] = eom
            records.append(vol[[self.id_col, self.eom_col, "uncertainty"]])

        if not records:
            return weights[[self.id_col, self.eom_col]].assign(uncertainty=self.uncertainty_floor)

        uncertainty = pd.concat(records, ignore_index=True)
        uncertainty = uncertainty.drop_duplicates(subset=[self.id_col, self.eom_col], keep="last")
        return uncertainty

    @staticmethod
    def _coerce_frame(obj: object, required: list[str], name: str) -> pd.DataFrame:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        missing = [col for col in required if col not in obj.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        return obj


@dataclass(frozen=True)
class IPCAUncertaintyDesign:
    """Design placeholder for the first unified benchmark path.

    This does not implement uncertainty yet. It simply records the intended
    structure for the first real joint model built on top of the validated IPCA
    benchmark.
    """

    benchmark_anchor: str = "ipca"
    uncertainty_family_candidates: tuple[str, ...] = (
        "exposure_instability",
        "factor_reconstruction_error",
        "rolling_residual_volatility",
    )
