# Uncertainty implementation fix plan

## Audit summary

Six issues found in the current uncertainty stack.
Three are high or medium severity and require code changes.
Three are lower severity and can be addressed incrementally.

---

## Issue 1 (high): TrailingResidualVolatilityModel computes total vol, not residual vol

### What is wrong

`_build_uncertainty_lookup` computes `ret_exc.std()` per stock per window.
This is total excess-return volatility — it includes the part the model explains.
Residual volatility should measure only the unexplained part:

```
residual_t = actual_ret_t - predicted_ret_t
residual_vol = std(residual_t)
```

### Why it matters

- High-beta or high-vol stocks get penalized regardless of model fit quality.
- A model that perfectly predicts a volatile stock still reports high uncertainty.
- The inverse-uncertainty overlay then under-weights exactly the stocks the model is most confident about, if those stocks happen to be volatile.
- This makes delta-vs-baseline comparisons uninterpretable: we cannot tell whether Sharpe changes come from uncertainty information or from variance tilting.

### Fix direction

**Option A — pass predicted returns into the uncertainty builder.**

Change `_build_uncertainty_lookup` to accept a `predicted_ret` DataFrame alongside `daily_ret`.
Compute `residual = actual - predicted` before taking the rolling std.

This requires the caller to supply predictions at daily or monthly granularity.
For LookupTable models that only have monthly weights, the simplest proxy:

1. From fitted monthly weights `w`, compute implied expected return per stock-month: `E[r]_i = w_i * k` where `k` is a cross-sectional scaling constant (or skip scaling and rank-center).
2. Compute monthly actual return from daily aggregation.
3. `residual_monthly = actual_monthly - E[r]_monthly`
4. Use rolling std of monthly residuals.

For models that produce daily predictions (e.g., tree models), pass daily predicted returns directly and compute daily residual vol.

**Option B — rename honestly and add a true residual model separately.**

- Rename current class to `TrailingTotalVolatilityModel` so the name matches what it does.
- Add a new `TrailingResidualVolatilityModel` that requires predicted returns.
- Keep both: total vol is a valid (if crude) uncertainty proxy; residual vol is the correct one.

**Recommendation: Option B first, then promote residual vol as the default.**

Option B is safer because:
- it does not break existing runner scripts or test expectations
- it makes the distinction between total and residual vol explicit in the API
- it lets us compare both in the same benchmark run to validate that residual vol is actually more informative

### Implementation sketch

```python
@dataclass
class TrailingTotalVolatilityModel(LookupTableJointModel):
    """Trailing stock-level TOTAL return volatility (renamed from residual)."""
    # current implementation, unchanged

@dataclass
class TrailingResidualVolatilityModel(LookupTableJointModel):
    """Trailing stock-level RESIDUAL volatility after removing model prediction."""

    def _build_uncertainty_lookup(self, weights, y):
        daily = self._coerce_frame(y, ...)
        # also require predicted returns — either from a new fit arg
        # or from weights -> implied signal
        # compute residuals, then rolling std
```

---

## Issue 2 (high): LookupTableJointModel is architecturally post-hoc

### What is wrong

The rebuild contract explicitly rejects post-hoc uncertainty attachment:
> "reject the old detached post-hoc q_hat pipeline as the active design"

But `LookupTableJointModel.fit()` takes **pre-computed weights** as input.
It does not fit a model. It wraps existing outputs with a sidecar uncertainty lookup.
The point prediction is frozen before uncertainty is computed.
Uncertainty cannot influence the point prediction.

This is post-hoc attachment behind a joint interface signature.

### Why it matters

- The "joint" interface is cosmetic — downstream code cannot tell whether uncertainty actually participated in fitting.
- If we report "joint model Sharpe" it is misleading: the point model never saw the uncertainty signal.
- New model families that plug into this interface will inherit the same flaw.

### Fix direction

**Phase 1 — be honest about the current adapter role.**

- Rename `LookupTableJointModel` to `PostHocUncertaintyAdapter` or `UncertaintyOverlayAdapter`.
- Keep the interface but document clearly: this is a transitional adapter, not the architectural target.
- Remove the docstring claim that this is "the package-level architectural target."

**Phase 2 — build a genuinely joint model for at least one family.**

The first real joint model should be IPCA, since that is the anchor.
In IPCA, factor exposures (betas) and factor returns (lambdas) are estimated jointly.
Natural joint uncertainty sources:

1. **Factor reconstruction error**: `X_hat = Gamma * F_t` vs actual characteristics. The reconstruction residual is intrinsic to the IPCA fitting — not post-hoc.
2. **Bootstrap instability of factor loadings**: re-fit IPCA with perturbed samples, measure loading dispersion. This is expensive but genuinely joint.
3. **Residual variance from the IPCA second-pass**: the IPCA model produces residuals as part of its fitting. Use these directly.

For tree/NN families, genuine joint uncertainty options:
- Quantile regression (predict median + IQR together)
- MC dropout (NN)
- Prediction interval from tree ensemble variance

**Phase 3 — demote LookupTable adapter to a convenience wrapper for backward compatibility.**

### Implementation priority

Phase 1 is a rename — do it immediately.
Phase 2 for IPCA is the first real deliverable.
Phase 2 for tree/NN can follow after IPCA is validated.

---

## Issue 3 (medium): "Exposure instability" misnomer

### What is wrong

`ArchivedIPCAExposureInstabilityModel` inherits `HistoricalWeightInstabilityModel`.
It measures the rolling std of portfolio weights `w` over time.
But IPCA "exposure" means factor loading (beta), not portfolio weight.

Portfolio weight = f(signal, factor loading, weighting rule).
Factor loading instability != portfolio weight instability.

### Fix

- Rename `ArchivedIPCAExposureInstabilityModel` -> `ArchivedIPCAWeightInstabilityModel`
- If actual factor-loading instability is desired, it must be computed from the IPCA model internals (Gamma matrix), not from output weights.
- Document the distinction in the contract boundaries doc.

---

## Issue 4 (medium): fit(X, y) signature semantic confusion

### What is wrong

The ABC declares `fit(X, y)` with numpy array type hints.
`LookupTableJointModel` overrides with `fit(X=weights_df, y=daily_df)`.
These are semantically incompatible: one is features/target, the other is pre-computed outputs/auxiliary data.

### Fix

**Option A — split the ABC into two protocols.**

```python
class JointUncertaintyModel(ABC):
    """For models that genuinely fit from features and targets."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self: ...
    def predict_with_uncertainty(self, X: np.ndarray) -> PredictionWithUncertainty: ...

class UncertaintyOverlayAdapter(ABC):
    """For post-hoc adapters that wrap pre-computed weights."""
    def fit(self, weights: pd.DataFrame, auxiliary: pd.DataFrame | None) -> Self: ...
    def predict_with_uncertainty(self, request: pd.DataFrame) -> PredictionWithUncertainty: ...
```

**Option B — use `object` type hints but enforce via runtime checks.**

Less clean but simpler. Not recommended long-term.

**Recommendation: Option A.** It makes the architectural distinction between genuine joint models and post-hoc adapters visible in the type system.

---

## Issue 5 (low): factor_reconstruction_error not implemented

### Status

Listed in the contract as one of three uncertainty candidates.
Not implemented anywhere in the codebase.

### Fix

Implement after Phase 2 of Issue 2.
Factor reconstruction error is the most natural IPCA-intrinsic uncertainty and should be the first genuinely joint uncertainty object.

Sketch:
- During IPCA fit, compute `X_hat = Z * Gamma_beta` (managed characteristics reconstruction)
- `recon_error_i = ||X_i - X_hat_i||` per stock per month
- This is available directly from the IPCA fitting output — no post-hoc computation needed.

---

## Issue 6 (low): inverse_uncertainty_overlay is too simplistic

### Status

Runner scripts use `w_new = w / uncertainty` then normalize.
This collapses to near-minimum-variance when uncertainty ~ total vol.

### Fix

Defer to after Issues 1-2 are resolved.
Once uncertainty is meaningful (residual-based, genuinely joint), the overlay function can be evaluated fairly.
Consider alternatives:
- Kelly-style scaling: `w_new = w * (expected_return / uncertainty^2)`
- Rank-based dampening: reduce weight rank by uncertainty rank
- Threshold gating: zero out weights where uncertainty exceeds a percentile cutoff

---

## Execution order

| Step | Issue | Action | Breaks existing code? |
|------|-------|--------|-----------------------|
| 1 | #3 | Rename exposure instability -> weight instability | Tests need update |
| 2 | #1 (Option B) | Rename current residual vol -> total vol, add true residual vol class | Tests need update |
| 3 | #4 | Split ABC into JointUncertaintyModel + UncertaintyOverlayAdapter | Imports change |
| 4 | #2 Phase 1 | Rename LookupTableJointModel -> UncertaintyOverlayAdapter subclass | Imports change |
| 5 | #2 Phase 2 | Implement first genuinely joint IPCA model using IPCA internals | New code |
| 6 | #5 | Implement factor reconstruction error as joint uncertainty source | New code |
| 7 | #6 | Evaluate and improve uncertainty -> weight overlay | Runner changes |

Steps 1-4 are renames/refactors — do them in one commit.
Steps 5-6 are new capability — separate commits.
Step 7 depends on 5-6 being validated.
