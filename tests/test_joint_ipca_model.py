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
