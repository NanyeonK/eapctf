"""Uncertainty method module.

Rebuilt active surface for interval / residual uncertainty methods.
Current scope is intentionally narrow:
- hold the uncertainty configuration contract
- provide a minimal overlay interface for already validated benchmark weights
- avoid broad uncertainty-system claims until benchmark parity is established
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UncertaintyConfig:
    """Minimal uncertainty configuration for the rebuild."""

    target: str = "oof_residual"
    representation: str = "interval_or_residual"
    fail_closed: bool = True


@dataclass(frozen=True)
class OverlayConfig:
    """Minimal uncertainty overlay on top of benchmark weights.

    `strength=0` leaves weights unchanged.
    """

    method: str = "inverse_interval"
    strength: float = 0.0
    floor_quantile: float = 0.05


def apply_uncertainty_overlay(
    base_weights: np.ndarray,
    uncertainty_score: np.ndarray,
    config: OverlayConfig | None = None,
) -> np.ndarray:
    """Apply a light uncertainty overlay to already validated benchmark weights.

    This function is intentionally conservative: it rescales existing weights
    rather than rebuilding the whole portfolio construction stack.
    """
    cfg = config or OverlayConfig()
    w = np.asarray(base_weights, dtype=float)
    u = np.asarray(uncertainty_score, dtype=float)
    if w.shape != u.shape:
        raise ValueError("base_weights and uncertainty_score must share shape")
    if cfg.strength == 0.0:
        return w.copy()

    floor = np.quantile(u, cfg.floor_quantile)
    floor = max(float(floor), 1e-8)
    u_safe = np.maximum(u, floor)

    if cfg.method == "inverse_interval":
        scale = 1.0 / u_safe
    elif cfg.method == "inverse_sqrt_interval":
        scale = 1.0 / np.sqrt(u_safe)
    elif cfg.method == "exp_neg_z":
        std = float(np.std(u_safe))
        z = (u_safe - float(np.mean(u_safe))) / (std if std > 1e-12 else 1.0)
        scale = np.exp(-cfg.strength * z)
        out = w * scale
        denom = np.sum(np.abs(out))
        return out / denom if denom > 1e-12 else np.zeros_like(out)
    else:
        raise ValueError(f"Unknown overlay method: {cfg.method}")

    scale = scale ** cfg.strength
    out = w * scale
    denom = np.sum(np.abs(out))
    return out / denom if denom > 1e-12 else np.zeros_like(out)
