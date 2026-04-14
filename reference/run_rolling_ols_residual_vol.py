#!/usr/bin/env python3
"""Run first non-IPCA adapter: rolling OLS baseline + residual-vol uncertainty."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from eapctf.ctf.baselines import RollingOLSBaseline
from eapctf.ctf.metrics import compute_metrics
from eapctf.ctf.uncertainty import TrailingResidualVolatilityModel

DATA_ROOT = Path.home() / "eap" / "data" / "ctf"
ARCHIVE_ROOT = Path.home() / "project_archive" / "eapctf_local_reset_20260414_165453" / "eapctf_before_rebuild"
IPCA_WEIGHTS = ARCHIVE_ROOT / "data" / "ctf" / "ipca_weights.parquet"
MAX_FEATURES = 20


@dataclass(frozen=True)
class OLSResidualVolReport:
    feature_count: int
    point_sharpe: float
    point_annual_return: float
    point_max_drawdown: float
    uncertainty_sharpe: float
    uncertainty_annual_return: float
    uncertainty_max_drawdown: float
    delta_vs_point: float
    ipca_anchor_sharpe: float
    delta_vs_ipca_anchor: float


def inverse_uncertainty_overlay(df: pd.DataFrame, floor: float) -> pd.DataFrame:
    out = df.copy()
    scale = 1.0 / out["uncertainty"].clip(lower=floor)
    out["w"] = out["w"] * scale
    out["w"] = out.groupby("eom")["w"].transform(lambda s: s / s.abs().sum())
    return out[["id", "eom", "w"]]


def main() -> None:
    chars = pd.read_parquet(DATA_ROOT / "ctff_chars.parquet")
    features = pd.read_parquet(DATA_ROOT / "ctff_features.parquet")
    daily = pd.read_parquet(DATA_ROOT / "ctff_daily_ret.parquet")
    ipca = pd.read_parquet(IPCA_WEIGHTS)

    feature_list = features["features"].tolist()[:MAX_FEATURES]
    baseline = RollingOLSBaseline(window_length=120, rank_transform=True)
    point_weights = baseline.run(chars, feature_list)

    joint_model = TrailingResidualVolatilityModel(model_family="ols", lookback_months=12, uncertainty_floor=1e-6)
    joint_model.fit(point_weights, daily)
    joint_frame = joint_model.predict_frame(point_weights[["id", "eom"]])
    uncertainty_weights = inverse_uncertainty_overlay(joint_frame, floor=joint_model.uncertainty_floor)

    point_metrics = compute_metrics(point_weights, daily)
    uncertainty_metrics = compute_metrics(uncertainty_weights, daily)
    ipca_metrics = compute_metrics(ipca[["id", "eom", "w"]], daily)

    report = OLSResidualVolReport(
        feature_count=len(feature_list),
        point_sharpe=float(point_metrics.sharpe),
        point_annual_return=float(point_metrics.annual_return),
        point_max_drawdown=float(point_metrics.max_drawdown),
        uncertainty_sharpe=float(uncertainty_metrics.sharpe),
        uncertainty_annual_return=float(uncertainty_metrics.annual_return),
        uncertainty_max_drawdown=float(uncertainty_metrics.max_drawdown),
        delta_vs_point=float(uncertainty_metrics.sharpe - point_metrics.sharpe),
        ipca_anchor_sharpe=float(ipca_metrics.sharpe),
        delta_vs_ipca_anchor=float(uncertainty_metrics.sharpe - ipca_metrics.sharpe),
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
