# Promoted family benchmark plan

Goal:
- move top smoke-pass families into a broader reduced-horizon benchmark stage on server2
- still test engineering robustness first, not draw paper conclusions yet

Selected families:
- `xgb`
- `lasso`
- `lgbm`
- `nn`

Why these four:
- they outperformed `ols`, `ridge`, and `rf` on the 12-month / 20-feature smoke stage
- they cover linear sparse, boosted-tree, and neural-network families

Benchmark stage defaults:
- train scheme: `expanding`
- features: 50
- test months: 60
- lasso tuning: 5-fold leakage-safe `forward_month_block` CV
- xgb/lgbm/nn: fixed cheap-but-broader configs suitable for reduced-horizon validation

Artifacts:
- server2 log: `/tmp/promoted_family_benchmark.log`
- runner: `reference/run_promoted_family_benchmark.py`

Success condition:
- all four families complete without crashes
- every family returns `id/eom/w` with valid metrics
- results are comparable to IPCA anchor on the same reduced horizon
