from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import pandas as pd

from eapctf.ctf import available_model_families, build_point_baseline, compute_metrics

DATA_ROOT = Path.home() / "eap" / "data" / "ctf"
ARCHIVE_ROOT = Path.home() / "project_archive" / "eapctf_local_reset_20260414_165453" / "eapctf_before_rebuild"
IPCA_WEIGHTS = ARCHIVE_ROOT / "data" / "ctf" / "ipca_weights.parquet"
LOG_PATH = Path(os.environ.get("EAPCTF_PROMOTED_LOG", "/tmp/promoted_family_benchmark.log"))
MAX_FEATURES = int(os.environ.get("EAPCTF_MAX_FEATURES", "50"))
MAX_TEST_MONTHS = int(os.environ.get("EAPCTF_MAX_TEST_MONTHS", "60"))
FAMILIES = os.environ.get("EAPCTF_FAMILIES", "xgb,lasso,lgbm,nn").split(",")


def emit(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def model_kwargs(name: str) -> dict:
    common = {
        "window_length": 120,
        "train_scheme": "expanding",
        "rank_transform": True,
    }
    if name == "lasso":
        return common | {"cv_folds": 5, "alpha_grid": [0.0001, 0.001, 0.01], "cv_scheme": "forward_month_block"}
    if name == "lgbm":
        return common | {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 31}
    if name == "xgb":
        return common | {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05}
    if name == "nn":
        return common | {"hidden_layer_sizes": (64, 32), "max_iter": 150}
    raise KeyError(name)


def main() -> None:
    LOG_PATH.write_text("", encoding="utf-8")
    families = [f.strip() for f in FAMILIES if f.strip()]
    unknown = [f for f in families if f not in available_model_families()]
    if unknown:
        raise ValueError(f"unknown model families: {unknown}")

    chars = pd.read_parquet(DATA_ROOT / "ctff_chars.parquet")
    features = pd.read_parquet(DATA_ROOT / "ctff_features.parquet")
    daily = pd.read_parquet(DATA_ROOT / "ctff_daily_ret.parquet")
    ipca = pd.read_parquet(IPCA_WEIGHTS)

    feature_list = features["features"].tolist()[:MAX_FEATURES]
    test_months = sorted(pd.to_datetime(chars.loc[chars["ctff_test"], "eom_ret"]).unique())
    selected_test_months = test_months[-MAX_TEST_MONTHS:]
    chars = chars[chars["eom_ret"].isin(selected_test_months) | (pd.to_datetime(chars["eom_ret"]) < selected_test_months[0])].copy()
    ipca = ipca[pd.to_datetime(ipca["eom"]).isin(selected_test_months)].copy()
    ipca_metrics = compute_metrics(ipca[["id", "eom", "w"]], daily)

    emit({
        "stage": "start",
        "families": families,
        "max_features": MAX_FEATURES,
        "max_test_months": MAX_TEST_MONTHS,
        "ipca_anchor_sharpe": float(ipca_metrics.sharpe),
    })

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
                "feature_count": len(feature_list),
                "test_months": MAX_TEST_MONTHS,
                "n_rows": int(len(weights)),
                "sharpe": float(metrics.sharpe),
                "annual_return": float(metrics.annual_return),
                "max_drawdown": float(metrics.max_drawdown),
                "delta_vs_ipca_anchor": float(metrics.sharpe - ipca_metrics.sharpe),
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
