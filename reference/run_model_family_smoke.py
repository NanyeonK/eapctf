from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import pandas as pd

from eapctf.ctf import available_model_families, build_point_baseline, compute_metrics

DATA_ROOT = Path.home() / "eap" / "data" / "ctf"
MAX_FEATURES = int(os.environ.get("EAPCTF_MAX_FEATURES", "20"))
MAX_TEST_MONTHS = int(os.environ.get("EAPCTF_MAX_TEST_MONTHS", "12"))
LOG_PATH = Path(os.environ.get("EAPCTF_SMOKE_LOG", "/tmp/model_family_smoke.log"))


def model_kwargs(name: str) -> dict:
    common = {
        "window_length": 120,
        "train_scheme": "expanding",
        "rank_transform": True,
    }
    if name == "ols":
        return common
    if name == "ridge":
        return common | {"cv_folds": 5, "alpha_grid": [0.0001, 0.001, 0.01]}
    if name == "lasso":
        return common | {"cv_folds": 5, "alpha_grid": [0.0001, 0.001, 0.01]}
    if name == "elastic_net":
        return common | {"cv_folds": 5, "alpha_grid": [0.0001, 0.001, 0.01], "l1_ratio_grid": [0.2, 0.5]}
    if name == "rf":
        return common | {"n_estimators": 50, "max_depth": 6}
    if name == "lgbm":
        return common | {"n_estimators": 80, "learning_rate": 0.05, "num_leaves": 31}
    if name == "xgb":
        return common | {"n_estimators": 80, "max_depth": 6, "learning_rate": 0.05}
    if name == "nn":
        return common | {"hidden_layer_sizes": (32,), "max_iter": 80}
    raise KeyError(name)


def emit(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    LOG_PATH.write_text("", encoding="utf-8")
    chars = pd.read_parquet(DATA_ROOT / "ctff_chars.parquet")
    features = pd.read_parquet(DATA_ROOT / "ctff_features.parquet")
    daily = pd.read_parquet(DATA_ROOT / "ctff_daily_ret.parquet")

    feature_list = features["features"].tolist()[:MAX_FEATURES]
    test_months = sorted(pd.to_datetime(chars.loc[chars["ctff_test"], "eom_ret"]).unique())
    selected_test_months = test_months[-MAX_TEST_MONTHS:]
    chars = chars[chars["eom_ret"].isin(selected_test_months) | (pd.to_datetime(chars["eom_ret"]) < selected_test_months[0])].copy()

    families = available_model_families()
    emit({"stage": "start", "families": families, "max_features": MAX_FEATURES, "max_test_months": MAX_TEST_MONTHS})

    for i, name in enumerate(families, start=1):
        try:
            baseline = build_point_baseline(name, **model_kwargs(name))
            weights = baseline.run(chars, feature_list)
            metrics = compute_metrics(weights, daily)
            emit({
                "stage": "result",
                "i": i,
                "total": len(families),
                "name": name,
                "status": "ok",
                "n_rows": int(len(weights)),
                "sharpe": float(metrics.sharpe),
                "annual_return": float(metrics.annual_return),
                "max_drawdown": float(metrics.max_drawdown),
                "selected_params_example": next(iter(baseline.selected_params_.values())) if baseline.selected_params_ else None,
            })
        except Exception as e:
            emit({
                "stage": "result",
                "i": i,
                "total": len(families),
                "name": name,
                "status": "error",
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(limit=5),
            })

    emit({"stage": "done", "log_path": str(LOG_PATH)})


if __name__ == "__main__":
    main()
