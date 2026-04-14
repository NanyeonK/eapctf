#!/usr/bin/env python3
"""Check archived IPCA parity against archived leaderboard references.

Uses:
- archived IPCA weights artifact
- current CTF daily returns file
- rebuilt `compute_metrics`
- archived documented leaderboard/reference values
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import pandas as pd

from eapctf.ctf.metrics import compute_metrics

ARCHIVE_ROOT = os.path.expanduser(
    "~/project_archive/eapctf_local_reset_20260414_165453/eapctf_before_rebuild"
)
ARCHIVED_IPCA_WEIGHTS = os.path.join(ARCHIVE_ROOT, "data/ctf/ipca_weights.parquet")
CURRENT_DAILY_RET = os.path.expanduser("~/eap/data/ctf/ctff_daily_ret.parquet")

# Archived documentation values from the pre-reset package docs
ARCHIVED_LOCAL_SHARPE = 1.939
ARCHIVED_LEADERBOARD_SHARPE = 1.948


@dataclass(frozen=True)
class IPCAParityReport:
    local_sharpe_now: float
    archived_local_sharpe: float
    archived_leaderboard_sharpe: float
    gap_vs_archived_local: float
    gap_vs_archived_leaderboard: float
    annual_return: float
    volatility: float
    max_drawdown: float
    benchmark_sharpe: float | None
    n_months: int


def main() -> None:
    weights = pd.read_parquet(ARCHIVED_IPCA_WEIGHTS)
    daily = pd.read_parquet(CURRENT_DAILY_RET)
    metrics = compute_metrics(weights, daily)
    report = IPCAParityReport(
        local_sharpe_now=float(metrics.sharpe),
        archived_local_sharpe=ARCHIVED_LOCAL_SHARPE,
        archived_leaderboard_sharpe=ARCHIVED_LEADERBOARD_SHARPE,
        gap_vs_archived_local=float(metrics.sharpe - ARCHIVED_LOCAL_SHARPE),
        gap_vs_archived_leaderboard=float(metrics.sharpe - ARCHIVED_LEADERBOARD_SHARPE),
        annual_return=float(metrics.annual_return),
        volatility=float(metrics.volatility),
        max_drawdown=float(metrics.max_drawdown),
        benchmark_sharpe=float(metrics.benchmark_sharpe) if metrics.benchmark_sharpe is not None else None,
        n_months=int(metrics.n_months),
    )
    print(asdict(report))


if __name__ == "__main__":
    main()
