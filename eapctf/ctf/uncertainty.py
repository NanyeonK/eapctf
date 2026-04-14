"""Uncertainty method bootstrap module.

This module replaces the old active naming surface that referred to
`conformal.py`. The rebuilt package will treat uncertainty estimation as a
separate design problem rather than assuming conformal semantics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UncertaintyConfig:
    """Minimal uncertainty configuration placeholder for the rebuild."""

    target: str = "oof_residual"
    representation: str = "interval_or_residual"
    fail_closed: bool = True
