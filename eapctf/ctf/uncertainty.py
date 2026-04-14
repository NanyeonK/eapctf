"""Uncertainty method module.

Active rebuild contract:
- point prediction and uncertainty must be emitted together
- post-hoc two-stage `q_hat` attachment is rejected as the primary design
- benchmark path should call one unified model interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


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
    def fit(self, X: np.ndarray, y: np.ndarray) -> "JointUncertaintyModel":
        """Fit a model that will later emit both point and uncertainty."""

    @abstractmethod
    def predict_with_uncertainty(self, X: np.ndarray) -> PredictionWithUncertainty:
        """Return point prediction and stock-level uncertainty together."""


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
