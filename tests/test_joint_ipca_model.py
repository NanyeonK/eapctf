from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eapctf.ctf.uncertainty import PredictionWithUncertainty


def test_prediction_with_uncertainty_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share shape"):
        PredictionWithUncertainty(
            point=np.array([0.1, 0.2]),
            uncertainty=np.array([0.3]),
        )


def test_lookup_table_joint_model_is_abstract() -> None:
    from eapctf.ctf.uncertainty import LookupTableJointModel

    with pytest.raises(TypeError):
        LookupTableJointModel(model_family="ols")


def test_rolling_ols_baseline_produces_test_period_weights() -> None:
    from eapctf.ctf.baselines import RollingOLSBaseline

    chars = pd.DataFrame(
        {
            "id": [1, 2, 1, 2, 1, 2],
            "eom": [
                "2020-01-31",
                "2020-01-31",
                "2020-02-29",
                "2020-02-29",
                "2020-03-31",
                "2020-03-31",
            ],
            "eom_ret": [
                "2020-01-31",
                "2020-01-31",
                "2020-02-29",
                "2020-02-29",
                "2020-03-31",
                "2020-03-31",
            ],
            "ret_exc_lead1m": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03],
            "ctff_test": [False, False, False, False, True, True],
            "f1": [1.0, -1.0, 1.2, -1.2, 1.1, -1.1],
            "f2": [0.5, -0.5, 0.6, -0.6, 0.55, -0.55],
        }
    )
    features = ["f1", "f2"]

    model = RollingOLSBaseline(window_length=2, min_train_rows=3, rank_transform=False)
    weights = model.run(chars, features)

    assert list(weights.columns) == ["id", "eom", "w"]
    assert len(weights) == 2
    assert set(weights["id"]) == {1, 2}
    assert set(pd.to_datetime(weights["eom"])) == {pd.Timestamp("2020-03-31")}
    assert np.isfinite(weights["w"]).all()


def test_expanding_ols_baseline_uses_all_available_history() -> None:
    from eapctf.ctf.baselines import RollingOLSBaseline

    chars = pd.DataFrame(
        {
            "id": [1, 2] * 4,
            "eom": [
                "2020-01-31", "2020-01-31",
                "2020-02-29", "2020-02-29",
                "2020-03-31", "2020-03-31",
                "2020-04-30", "2020-04-30",
            ],
            "eom_ret": [
                "2020-01-31", "2020-01-31",
                "2020-02-29", "2020-02-29",
                "2020-03-31", "2020-03-31",
                "2020-04-30", "2020-04-30",
            ],
            "ret_exc_lead1m": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.04, -0.04],
            "ctff_test": [False, False, False, False, True, True, True, True],
            "f1": [1.0, -1.0, 1.1, -1.1, 1.2, -1.2, 1.3, -1.3],
            "f2": [0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7],
        }
    )
    features = ["f1", "f2"]

    model = RollingOLSBaseline(
        train_scheme="expanding",
        window_length=1,
        min_train_rows=3,
        rank_transform=False,
    )
    weights = model.run(chars, features)

    assert set(pd.to_datetime(weights["eom"])) == {
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-04-30"),
    }
    assert model.last_train_row_count_ == 6


def test_forward_month_block_cv_never_trains_on_future_months() -> None:
    from eapctf.ctf.baselines import RollingRidgeBaseline

    chars = pd.DataFrame(
        {
            "id": [1, 2] * 6,
            "eom": [
                "2020-01-31", "2020-01-31",
                "2020-02-29", "2020-02-29",
                "2020-03-31", "2020-03-31",
                "2020-04-30", "2020-04-30",
                "2020-05-31", "2020-05-31",
                "2020-06-30", "2020-06-30",
            ],
            "eom_ret": [
                "2020-01-31", "2020-01-31",
                "2020-02-29", "2020-02-29",
                "2020-03-31", "2020-03-31",
                "2020-04-30", "2020-04-30",
                "2020-05-31", "2020-05-31",
                "2020-06-30", "2020-06-30",
            ],
            "ret_exc_lead1m": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.04, -0.04, 0.05, -0.05, 0.06, -0.06],
            "ctff_test": [False] * 10 + [True, True],
            "f1": [1.0, -1.0, 1.1, -1.1, 1.2, -1.2, 1.3, -1.3, 1.4, -1.4, 1.5, -1.5],
            "f2": [0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7, 0.8, -0.8],
        }
    )
    features = ["f1", "f2"]
    model = RollingRidgeBaseline(
        train_scheme="expanding",
        window_length=5,
        min_train_rows=8,
        rank_transform=False,
        cv_folds=5,
        cv_scheme="forward_month_block",
        alpha_grid=[0.0001, 0.01],
    )
    train = model._slice_training_window(chars, pd.Timestamp("2020-06-30"), features)
    x_train = train[features].to_numpy(dtype=float)
    y_train = train["ret_exc_lead1m"].to_numpy(dtype=float)
    eoms = pd.to_datetime(train["eom"]).to_numpy()

    splits = model._cv_split_indices(x_train, y_train, eoms)

    assert len(splits) >= 2
    for train_idx, val_idx in splits:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert pd.Timestamp(eoms[train_idx].max()) < pd.Timestamp(eoms[val_idx].min())


def test_regularized_linear_baselines_support_5fold_cv() -> None:
    from eapctf.ctf.baselines import (
        RollingElasticNetBaseline,
        RollingLassoBaseline,
        RollingRidgeBaseline,
    )

    chars = pd.DataFrame(
        {
            "id": [1, 2, 3] * 4,
            "eom": [
                "2020-01-31", "2020-01-31", "2020-01-31",
                "2020-02-29", "2020-02-29", "2020-02-29",
                "2020-03-31", "2020-03-31", "2020-03-31",
                "2020-04-30", "2020-04-30", "2020-04-30",
            ],
            "eom_ret": [
                "2020-01-31", "2020-01-31", "2020-01-31",
                "2020-02-29", "2020-02-29", "2020-02-29",
                "2020-03-31", "2020-03-31", "2020-03-31",
                "2020-04-30", "2020-04-30", "2020-04-30",
            ],
            "ret_exc_lead1m": [0.03, 0.00, -0.03, 0.04, 0.00, -0.04, 0.05, 0.00, -0.05, 0.06, 0.00, -0.06],
            "ctff_test": [False, False, False, False, False, False, False, False, False, True, True, True],
            "f1": [1.0, 0.0, -1.0, 1.1, 0.0, -1.1, 1.2, 0.0, -1.2, 1.3, 0.0, -1.3],
            "f2": [0.6, 0.0, -0.6, 0.7, 0.0, -0.7, 0.8, 0.0, -0.8, 0.9, 0.0, -0.9],
        }
    )
    features = ["f1", "f2"]
    alpha_grid = [0.0001, 0.01, 0.1]

    models = [
        RollingRidgeBaseline(window_length=3, min_train_rows=9, rank_transform=False, cv_folds=5, alpha_grid=alpha_grid, cv_scheme="forward_month_block"),
        RollingLassoBaseline(window_length=3, min_train_rows=9, rank_transform=False, cv_folds=5, alpha_grid=alpha_grid, cv_scheme="forward_month_block"),
        RollingElasticNetBaseline(window_length=3, min_train_rows=9, rank_transform=False, cv_folds=5, alpha_grid=alpha_grid, l1_ratio_grid=[0.2, 0.5], cv_scheme="forward_month_block"),
    ]

    for model in models:
        weights = model.run(chars, features)
        assert len(weights) == 3
        assert np.isfinite(weights["w"]).all(), model
        assert len(model.selected_params_) == 1
        params = next(iter(model.selected_params_.values()))
        assert params["alpha"] in alpha_grid


def test_generic_joint_model_supports_non_ipca_model_family() -> None:
    from eapctf.ctf.uncertainty import HistoricalWeightInstabilityModel

    weights = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "eom": ["2020-01-31", "2020-02-29", "2020-03-31"],
            "w": [0.1, 0.2, 0.3],
        }
    )

    model = HistoricalWeightInstabilityModel(
        model_family="ols",
        lookback_months=2,
        uncertainty_floor=1e-6,
    )
    model.fit(weights, None)
    result = model.predict_with_uncertainty(weights[["id", "eom"]])

    assert model.model_family == "ols"
    np.testing.assert_allclose(result.point, weights["w"].to_numpy())
    np.testing.assert_allclose(result.uncertainty[0], 1e-6)
    np.testing.assert_allclose(result.uncertainty[2], np.std([0.1, 0.2], ddof=1))


def test_generic_residual_volatility_model_supports_non_ipca_model_family() -> None:
    from eapctf.ctf.uncertainty import TrailingResidualVolatilityModel

    weights = pd.DataFrame(
        {
            "id": [1, 2],
            "eom": ["2020-01-31", "2020-01-31"],
            "w": [0.25, -0.15],
        }
    )
    daily = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "date": ["2020-01-10", "2020-01-15", "2020-01-12", "2020-01-14"],
            "ret_exc": [0.02, -0.02, 0.01, 0.03],
        }
    )

    model = TrailingResidualVolatilityModel(
        model_family="tree_boosting",
        lookback_months=1,
        uncertainty_floor=1e-6,
    )
    model.fit(weights, daily)
    frame = model.predict_frame(weights[["id", "eom"]])

    assert set(frame.columns) == {"id", "eom", "w", "uncertainty", "model_family"}
    assert set(frame["model_family"]) == {"tree_boosting"}
    np.testing.assert_allclose(frame["w"].to_numpy(), weights["w"].to_numpy())


def test_archived_ipca_joint_model_returns_point_weights_and_trailing_volatility() -> None:
    from eapctf.ctf.uncertainty import ArchivedIPCAResidualVolatilityModel

    weights = pd.DataFrame(
        {
            "id": [1, 2],
            "eom": ["2020-01-31", "2020-01-31"],
            "w": [0.2, -0.2],
        }
    )
    daily = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "date": [
                "2020-01-15",
                "2020-01-20",
                "2020-02-10",
                "2020-01-10",
                "2020-01-20",
            ],
            "ret_exc": [0.01, -0.01, 0.50, 0.02, 0.02],
        }
    )
    request = weights[["id", "eom"]].copy()

    model = ArchivedIPCAResidualVolatilityModel(lookback_months=1, uncertainty_floor=1e-6)
    model.fit(weights, daily)
    result = model.predict_with_uncertainty(request)

    np.testing.assert_allclose(result.point, np.array([0.2, -0.2]))
    np.testing.assert_allclose(result.uncertainty[0], np.std([0.01, -0.01], ddof=1))
    np.testing.assert_allclose(result.uncertainty[1], 1e-6)
    assert result.uncertainty[0] < 0.1, "future February return leaked into January uncertainty"


def test_archived_ipca_exposure_instability_uses_only_past_weights() -> None:
    from eapctf.ctf.uncertainty import ArchivedIPCAExposureInstabilityModel

    weights = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "eom": [
                "2020-01-31",
                "2020-02-29",
                "2020-03-31",
                "2020-01-31",
                "2020-02-29",
                "2020-03-31",
            ],
            "w": [0.10, 0.30, 1.20, -0.10, -0.10, -0.40],
        }
    )
    request = weights[["id", "eom"]].copy()

    model = ArchivedIPCAExposureInstabilityModel(lookback_months=2, uncertainty_floor=1e-6)
    model.fit(weights, None)
    result = model.predict_with_uncertainty(request)

    np.testing.assert_allclose(result.point, weights["w"].to_numpy())
    np.testing.assert_allclose(result.uncertainty[0], 1e-6)
    np.testing.assert_allclose(result.uncertainty[1], 1e-6)
    np.testing.assert_allclose(result.uncertainty[2], np.std([0.10, 0.30], ddof=1))
    np.testing.assert_allclose(result.uncertainty[5], 1e-6)
    assert result.uncertainty[2] < 1.0, "current-month 1.20 weight leaked into March instability"
