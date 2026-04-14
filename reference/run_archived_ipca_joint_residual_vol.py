#!/usr/bin/env python3
"""Run first unified IPCA benchmark object with trailing residual-vol uncertainty."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import pandas as pd

from eapctf.ctf.metrics import compute_metrics
from eapctf.ctf.uncertainty import ArchivedIPCAResidualVolatilityModel

ARCHIVE_ROOT = os.path.expanduser(
    "~/project_archive/eapctf_local_reset_20260414_165453/eapctf_before_rebuild"
)
ARCHIVED_IPCA_WEIGHTS = os.path.join(ARCHIVE_ROOT, "data/ctf/ipca_weights.parquet")
CURRENT_DAILY_RET = os.path.expanduser("~/eap/data/ctf/ctff_daily_ret.parquet")


@dataclass(frozen=True)
class JointIPCAResidualVolReport:
    benchmark_sharpe: float
    uncertainty_sharpe: float
    delta_sharpe: float
    benchmark_annual_return: float
    uncertainty_annual_return: float
    benchmark_max_drawdown: float
    uncertainty_max_drawdown: float
    uncertainty_floor: float
    lookback_months: int
    n_rows: int


def inverse_uncertainty_overlay(df: pd.DataFrame, floor: float) -> pd.DataFrame:
    out = df.copy()
    scale = 1.0 / out["uncertainty"].clip(lower=floor)
    out["w"] = out["w"] * scale
    out["w"] = out.groupby("eom")["w"].transform(lambda s: s / s.abs().sum())
    return out[["id", "eom", "w"]]


def main() -> None:
    weights = pd.read_parquet(ARCHIVED_IPCA_WEIGHTS)
    daily = pd.read_parquet(CURRENT_DAILY_RET)

    model = ArchivedIPCAResidualVolatilityModel(lookback_months=12, uncertainty_floor=1e-6)
    model.fit(weights, daily)
    joint = model.predict_frame(weights[["id", "eom"]])

    benchmark_metrics = compute_metrics(weights[["id", "eom", "w"]], daily)
    uncertainty_weights = inverse_uncertainty_overlay(joint, floor=model.uncertainty_floor)
    uncertainty_metrics = compute_metrics(uncertainty_weights, daily)

    report = JointIPCAResidualVolReport(
        benchmark_sharpe=float(benchmark_metrics.sharpe),
        uncertainty_sharpe=float(uncertainty_metrics.sharpe),
        delta_sharpe=float(uncertainty_metrics.sharpe - benchmark_metrics.sharpe),
        benchmark_annual_return=float(benchmark_metrics.annual_return),
        uncertainty_annual_return=float(uncertainty_metrics.annual_return),
        benchmark_max_drawdown=float(benchmark_metrics.max_drawdown),
        uncertainty_max_drawdown=float(uncertainty_metrics.max_drawdown),
        uncertainty_floor=model.uncertainty_floor,
        lookback_months=model.lookback_months,
        n_rows=int(len(joint)),
    )
    print(json.dumps(asdict(report), indent=2, default=str))


if __name__ == "__main__":
    main()
