# eapctf rebuild notes

The old active uncertainty module name `conformal.py` is retired.
The rebuilt package starts from `eapctf/ctf/uncertainty.py`.

Migration rule:
- do not create a new active `eapctf/ctf/conformal.py`
- if old logic is recovered from backups, port only the needed pieces into
  the new uncertainty stack after review

## Linear-model CV note

Regularized linear baselines now should prefer sklearn-native CV estimators
under leakage-safe month-aware forward splits:
- `RidgeCV`
- `LassoCV`
- `ElasticNetCV`

Implementation rule:
- do not use row-level random CV by default for backtest-oriented tuning
- default to `forward_month_block` splits on unique month values
- keep `random_kfold` only as an explicit auxiliary option

## Fast model-family guidance

When adding heavier model families, default speed strategy should be:

### RF
- use `RandomForestRegressor(n_jobs=-1)`
- tune coarse grid only
- reduce refit frequency first before expanding grid
- parallelize over spec/job level, not inner Python loops only

### XGB
- use histogram tree method / approximate tree builder
- use early stopping on a month-aware validation block
- keep depth and boosting rounds conservative for screening
- note: `xgboost` is currently missing from the server2 venv, so add as an optional dependency before enabling

### LGBM
- use LightGBM native multithreading (`n_jobs=-1`)
- prefer leaf/feature subsampling and early stopping
- keep a fixed small screening grid before any large CV sweep
- package is available in server2 venv (`lightgbm 4.6.0`)

### NN
- use mini-batch training, explicit epoch caps, and early stopping
- cache tensors / preprocessed arrays once per train window when possible
- first screen with shallow architectures only
- if GPU is used later, compare dataloader overhead vs actual gain before committing
- `torch 2.5.1+cu121` is available in server2 venv

### General speed rule
- prefer sklearn/native estimator CV helpers over manual nested fit loops when available
- parallelize at spec/job level if a single matrix run becomes bottlenecked by one slow family
- screening stage first, promotion stage second
- only promising families get long-horizon or full-period runs
