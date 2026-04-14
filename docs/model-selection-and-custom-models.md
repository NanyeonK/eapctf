# Model selection and custom model extension

This package now supports point-baseline model selection through a registry.

## Built-in model families
- `ols`
- `ridge`
- `lasso`
- `elastic_net`
- `rf`
- `lgbm`
- `xgb`
- `nn`

## Public registry API
Location:
- `eapctf.ctf.model_registry`

Functions:
- `available_model_families()`
- `build_point_baseline(name, **kwargs)`
- `register_model(name, cls)`

## Contract for custom models
A custom point-baseline class should follow the existing baseline surface.

Minimum practical contract:
- constructor accepts baseline config fields such as `window_length`, `train_scheme`, `rank_transform`
- `.run(chars, features) -> DataFrame`
- output columns exactly:
  - `id`
  - `eom`
  - `w`

Recommended implementation path:
- subclass the shared baseline class in `eapctf/ctf/baselines.py`
- implement `fit_estimator(...)`
- reuse existing rolling/expanding data slicing and weight construction

## Example: register a custom model
```python
from eapctf.ctf.baselines import RollingOLSBaseline
from eapctf.ctf.model_registry import register_model, build_point_baseline

class MyCustomOLS(RollingOLSBaseline):
    model_family = "custom_ols"

register_model("custom_ols", MyCustomOLS)
model = build_point_baseline("custom_ols", window_length=120, train_scheme="expanding")
```

## Smoke-test checklist for new models
- import works on server2
- reduced-horizon smoke run completes
- output columns are exactly `id/eom/w`
- weights are finite
- training uses only `eom_ret < test_date`
- if CV is used, split scheme is explicit and leakage-safe

## Notes on heavy model families
- `rf`: use `n_jobs=-1`, coarse grids first
- `lgbm`: use native multithreading and small screening grids first
- `xgb`: use `tree_method="hist"` and cheap configs first
- `nn`: use small architectures and low epoch caps for smoke stage
