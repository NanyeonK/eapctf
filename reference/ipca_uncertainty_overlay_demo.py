#!/usr/bin/env python3
"""Minimal IPCA uncertainty overlay demo.

Uses archived IPCA weights as validated benchmark input and applies a very light
uncertainty overlay only after parity has been checked.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from eapctf.ctf.metrics import compute_metrics
from eapctf.ctf.uncertainty import OverlayConfig, apply_uncertainty_overlay

ARCHIVE_ROOT = os.path.expanduser(
    "~/project_archive/eapctf_local_reset_20260414_165453/eapctf_before_rebuild"
)
ARCHIVED_IPCA_WEIGHTS = os.path.join(ARCHIVE_ROOT, "data/ctf/ipca_weights.parquet")
CURRENT_DAILY_RET = os.path.expanduser("~/eap/data/ctf/ctff_daily_ret.parquet")


def main() -> None:
    weights = pd.read_parquet(ARCHIVED_IPCA_WEIGHTS)
    # Placeholder uncertainty proxy: within-month absolute benchmark weight rank.
    # Replace with a real uncertainty score only after the package uncertainty
    # contract is rebuilt.
    overlay_frames = []
    for eom, g in weights.groupby("eom", sort=False):
        u = np.abs(g["w"].to_numpy())
        w = g["w"].to_numpy()
        new_w = apply_uncertainty_overlay(
            w,
            u,
            OverlayConfig(method="inverse_sqrt_interval", strength=0.15),
        )
        tmp = g.copy()
        tmp["w"] = new_w
        overlay_frames.append(tmp)
    overlay = pd.concat(overlay_frames, ignore_index=True)
    daily = pd.read_parquet(CURRENT_DAILY_RET)
    base_metrics = compute_metrics(weights, daily)
    overlay_metrics = compute_metrics(overlay, daily)
    print({
        "base_sharpe": float(base_metrics.sharpe),
        "overlay_sharpe": float(overlay_metrics.sharpe),
        "delta_sharpe": float(overlay_metrics.sharpe - base_metrics.sharpe),
        "note": "demo only; current uncertainty proxy is placeholder based on abs(weight)",
    })


if __name__ == "__main__":
    main()
