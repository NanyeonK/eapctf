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
