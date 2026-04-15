"""Uncertainty method module.

Active rebuild contract:
- point prediction and uncertainty must be emitted together at inference time
- genuinely joint models and post-hoc uncertainty overlays are distinct concepts
- overlay adapters remain available as transitional wrappers for fixed point baselines
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UncertaintyConfig:
    """Configuration for point + uncertainty prediction objects."""

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
    """For genuinely joint models fit from features and targets."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "JointUncertaintyModel":
        """Fit a model that jointly defines point prediction and uncertainty."""

    @abstractmethod
    def predict_with_uncertainty(self, X: np.ndarray) -> PredictionWithUncertainty:
        """Return point prediction and uncertainty from one fitted model."""


class UncertaintyOverlayAdapter(ABC):
    """For post-hoc uncertainty overlays on pre-computed point baselines.

    This is transitional adapter surface. Point weights are pre-computed, then an
    uncertainty lookup is attached for evaluation or overlay experiments.
    """

    @abstractmethod
    def fit(self, weights: pd.DataFrame, auxiliary: object) -> "UncertaintyOverlayAdapter":
        """Attach uncertainty lookup to pre-computed point weights."""

    @abstractmethod
    def predict_with_uncertainty(self, request: pd.DataFrame) -> PredictionWithUncertainty:
        """Return stored point weights and attached uncertainty for request rows."""


@dataclass
class LookupTableUncertaintyAdapter(UncertaintyOverlayAdapter):
    """Shared lookup-table adapter for post-hoc uncertainty overlays.

    This is not the permanent architecture target. It wraps a fixed point-output
    table `(id, eom, w)` and supplies uncertainty from an auxiliary lookup.
    """

    model_family: str
    lookback_months: int = 12
    uncertainty_floor: float = 1e-6
    id_col: str = "id"
    eom_col: str = "eom"
    w_col: str = "w"
    date_col: str = "date"
    ret_col: str = "ret_exc"
    _point_lookup: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _uncertainty_lookup: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def fit(self, weights: pd.DataFrame, auxiliary: object) -> "LookupTableUncertaintyAdapter":
        weights = self._coerce_frame(weights, required=[self.id_col, self.eom_col, self.w_col], name="weights")
        weights = weights[[self.id_col, self.eom_col, self.w_col]].copy()
        weights[self.eom_col] = pd.to_datetime(weights[self.eom_col])
        weights = weights.sort_values([self.eom_col, self.id_col]).reset_index(drop=True)

        self._point_lookup = weights
        self._uncertainty_lookup = self._build_uncertainty_lookup(weights, auxiliary)
        return self

    def predict_with_uncertainty(self, request: pd.DataFrame) -> PredictionWithUncertainty:
        request = self._coerce_frame(request, required=[self.id_col, self.eom_col], name="request")
        if self._point_lookup is None or self._uncertainty_lookup is None:
            raise RuntimeError("adapter must be fit before prediction")

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

    def predict_frame(self, request: pd.DataFrame) -> pd.DataFrame:
        request = self._coerce_frame(request, required=[self.id_col, self.eom_col], name="request")
        request = request[[self.id_col, self.eom_col]].copy()
        request[self.eom_col] = pd.to_datetime(request[self.eom_col])
        prediction = self.predict_with_uncertainty(request)
        out = request.copy()
        out[self.w_col] = prediction.point
        out["uncertainty"] = prediction.uncertainty
        out["model_family"] = self.model_family
        return out

    @abstractmethod
    def _build_uncertainty_lookup(self, weights: pd.DataFrame, auxiliary: object) -> pd.DataFrame:
        """Build `(id, eom, uncertainty)` lookup for fixed point baseline."""

    @staticmethod
    def _coerce_frame(obj: object, required: list[str], name: str) -> pd.DataFrame:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        missing = [col for col in required if col not in obj.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        return obj


@dataclass
class TrailingTotalVolatilityModel(LookupTableUncertaintyAdapter):
    """Generic point baseline + trailing stock-level TOTAL return volatility.

    This is the honest rename of the old residual-vol proxy. It uses total stock
    return variability, not model residual variability.
    """

    def _build_uncertainty_lookup(self, weights: pd.DataFrame, auxiliary: object) -> pd.DataFrame:
        daily = self._coerce_frame(auxiliary, required=[self.id_col, self.date_col, self.ret_col], name="daily_ret")
        daily = daily[[self.id_col, self.date_col, self.ret_col]].copy()
        daily[self.date_col] = pd.to_datetime(daily[self.date_col])
        daily = daily.sort_values([self.date_col, self.id_col]).reset_index(drop=True)

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


@dataclass
class TrailingResidualVolatilityModel(LookupTableUncertaintyAdapter):
    """Generic point baseline + trailing RESIDUAL volatility.

    Auxiliary input must be a dict containing:
    - `daily_ret`: DataFrame with `id,date,ret_exc`
    - `predicted_ret`: DataFrame with `id,eom,predicted_ret`

    Residuals are computed at monthly horizon:
    actual_monthly_ret - predicted_ret.
    """

    predicted_col: str = "predicted_ret"

    def _build_uncertainty_lookup(self, weights: pd.DataFrame, auxiliary: object) -> pd.DataFrame:
        if not isinstance(auxiliary, dict):
            raise TypeError("auxiliary must be dict with daily_ret and predicted_ret")
        if "daily_ret" not in auxiliary or "predicted_ret" not in auxiliary:
            raise ValueError("auxiliary must contain daily_ret and predicted_ret")

        daily = self._coerce_frame(auxiliary["daily_ret"], required=[self.id_col, self.date_col, self.ret_col], name="daily_ret")
        predicted = self._coerce_frame(auxiliary["predicted_ret"], required=[self.id_col, self.eom_col, self.predicted_col], name="predicted_ret")

        daily = daily[[self.id_col, self.date_col, self.ret_col]].copy()
        daily[self.date_col] = pd.to_datetime(daily[self.date_col])
        predicted = predicted[[self.id_col, self.eom_col, self.predicted_col]].copy()
        predicted[self.eom_col] = pd.to_datetime(predicted[self.eom_col])

        weights = weights.copy()
        weights["_ret_month"] = weights[self.eom_col].dt.to_period("M") + 1
        daily["_ret_month"] = daily[self.date_col].dt.to_period("M")
        actual_monthly = (
            weights[[self.id_col, self.eom_col, "_ret_month"]]
            .merge(daily[[self.id_col, "_ret_month", self.ret_col]], on=[self.id_col, "_ret_month"], how="left")
            .groupby([self.id_col, self.eom_col])[self.ret_col]
            .sum()
            .rename("actual_ret")
            .reset_index()
        )

        monthly = actual_monthly.merge(predicted, on=[self.id_col, self.eom_col], how="inner")
        monthly["residual"] = monthly["actual_ret"] - monthly[self.predicted_col]
        monthly = monthly.sort_values([self.id_col, self.eom_col]).reset_index(drop=True)

        rows: list[dict[str, object]] = []
        for stock_id, group in monthly.groupby(self.id_col, sort=False):
            group = group.sort_values(self.eom_col).reset_index(drop=True)
            resid = group["residual"].astype(float)
            for idx, row in group.iterrows():
                start = max(0, idx - self.lookback_months)
                past = resid.iloc[start:idx]
                if len(past) < 2:
                    uncertainty = self.uncertainty_floor
                else:
                    uncertainty = float(past.std(ddof=1))
                    if not np.isfinite(uncertainty):
                        uncertainty = self.uncertainty_floor
                    uncertainty = max(uncertainty, self.uncertainty_floor)
                rows.append({
                    self.id_col: stock_id,
                    self.eom_col: row[self.eom_col],
                    "uncertainty": uncertainty,
                })
        return pd.DataFrame(rows)


@dataclass
class HistoricalWeightInstabilityModel(LookupTableUncertaintyAdapter):
    """Generic point baseline + trailing instability of past point weights."""

    min_history: int = 2

    def _build_uncertainty_lookup(self, weights: pd.DataFrame, auxiliary: object) -> pd.DataFrame:
        if auxiliary is not None:
            _ = auxiliary

        rows: list[dict[str, object]] = []
        for stock_id, group in weights.groupby(self.id_col, sort=False):
            group = group.sort_values(self.eom_col).reset_index(drop=True)
            history = group[self.w_col].astype(float)
            for idx, row in group.iterrows():
                start = max(0, idx - self.lookback_months)
                past = history.iloc[start:idx]
                if len(past) < self.min_history:
                    uncertainty = self.uncertainty_floor
                else:
                    uncertainty = float(past.std(ddof=1))
                    if not np.isfinite(uncertainty):
                        uncertainty = self.uncertainty_floor
                    uncertainty = max(uncertainty, self.uncertainty_floor)
                rows.append({
                    self.id_col: stock_id,
                    self.eom_col: row[self.eom_col],
                    "uncertainty": uncertainty,
                })
        return pd.DataFrame(rows)


@dataclass
class ArchivedIPCATotalVolatilityModel(TrailingTotalVolatilityModel):
    """IPCA anchor adapter for total-volatility overlay proxy."""

    model_family: str = "ipca"


@dataclass
class ArchivedIPCAWeightInstabilityModel(HistoricalWeightInstabilityModel):
    """IPCA anchor adapter for historical point-weight instability."""

    model_family: str = "ipca"


@dataclass(frozen=True)
class IPCAUncertaintyDesign:
    """Design placeholder for first audited rebuild anchor."""

    benchmark_anchor: str = "ipca"
    uncertainty_family_candidates: tuple[str, ...] = (
        "weight_instability",
        "factor_reconstruction_error",
        "trailing_total_volatility",
        "trailing_residual_volatility",
    )
    architecture_target: str = "genuine_joint_models_plus_explicit_overlay_adapters"
