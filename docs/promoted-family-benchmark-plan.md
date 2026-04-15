# Promoted family benchmark plan

Goal:
- move top smoke-pass families into a broader benchmark-like reduced analysis path on server2
- keep engineering validation first, but make the default configuration align with the benchmark surface rather than a narrow smoke override

Selected families:
- `xgb`
- `lasso`
- `lgbm`
- `nn`

Why these four:
- they outperformed `ols`, `ridge`, and `rf` on the first 12-month / 20-feature smoke stage
- they cover sparse linear, boosted-tree, and neural-network families

Benchmark-like defaults:
- train scheme: `expanding`
- features: all available benchmark features by default
- test months: all available benchmark test months by default
- lasso tuning: 5-fold leakage-safe `forward_month_block` CV
- xgb/lgbm/nn: fixed configs appropriate for first benchmark-like pass

Reduced smoke overrides remain available only through env vars:
- `EAPCTF_MAX_FEATURES`
- `EAPCTF_MAX_TEST_MONTHS`

Artifacts:
- server2 log: `/tmp/promoted_family_benchmark.log`
- runner: `reference/run_promoted_family_benchmark.py`

Success condition:
- all four families complete without crashes under the benchmark-like default config
- every family returns valid `id/eom/w`
- results are directly comparable to the slice-matched IPCA anchor under the same default horizon
