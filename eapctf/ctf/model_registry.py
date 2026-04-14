from __future__ import annotations

from collections.abc import Callable
from typing import Any

from eapctf.ctf.baselines import (
    RollingElasticNetBaseline,
    RollingLassoBaseline,
    RollingLightGBMBaseline,
    RollingMLPBaseline,
    RollingOLSBaseline,
    RollingRandomForestBaseline,
    RollingRidgeBaseline,
    RollingXGBoostBaseline,
)

MODEL_REGISTRY: dict[str, type] = {
    "ols": RollingOLSBaseline,
    "ridge": RollingRidgeBaseline,
    "lasso": RollingLassoBaseline,
    "elastic_net": RollingElasticNetBaseline,
    "rf": RollingRandomForestBaseline,
    "lgbm": RollingLightGBMBaseline,
    "xgb": RollingXGBoostBaseline,
    "nn": RollingMLPBaseline,
}


def available_model_families() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def register_model(name: str, cls: type) -> None:
    MODEL_REGISTRY[name] = cls


def build_point_baseline(name: str, **kwargs: Any):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model family: {name}")
    return MODEL_REGISTRY[name](**kwargs)
