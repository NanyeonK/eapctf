# Model family expansion implementation plan

> For Hermes: execute in small TDD-first steps. This is a smoke-test stage, not a final research benchmark stage.

Goal: add point-baseline model selection for RF, LGBM, XGB, and NN on top of the existing rolling/expanding baseline framework, and document how future custom models can plug into the same registry.

Architecture:
- Keep `eapctf/ctf/baselines.py` as the shared point-baseline implementation surface.
- Add non-linear baseline classes that reuse the existing rolling/expanding dataset slicing and weight-construction path.
- Add a simple registry in `eapctf/ctf/model_registry.py` so smoke tests and future runners can select model families by string name.
- Add a server2 smoke runner that executes all registered families on a reduced horizon with cheap configs and writes a JSONL progress log.

Tech stack:
- scikit-learn: RF, MLP, linear baselines
- lightgbm: LGBM baseline
- xgboost: XGB baseline
- pandas/numpy: panel prep and weights

---

## Task 1: Add failing tests for requested model families and registry

Files:
- Modify: `tests/test_joint_ipca_model.py`

Step 1: Add tests that require:
- registry includes `rf`, `lgbm`, `xgb`, `nn`
- `RollingRandomForestBaseline`, `RollingLightGBMBaseline`, `RollingXGBoostBaseline`, `RollingMLPBaseline` all produce finite `id/eom/w` outputs on a tiny synthetic panel
- custom model registration works through a registry API

Step 2: Run tests to verify failure
Run:
- `PYTHONPATH=~/project/eapctf ~/eap/.venv/bin/python -m pytest tests/test_joint_ipca_model.py -q`
Expected:
- import / registry failures before implementation

## Task 2: Implement non-linear point baseline classes

Files:
- Modify: `eapctf/ctf/baselines.py`

Objective:
- add RF/LGBM/XGB/NN baselines without changing the already-tested linear path semantics

Classes to add:
- `RollingRandomForestBaseline`
- `RollingLightGBMBaseline`
- `RollingXGBoostBaseline`
- `RollingMLPBaseline`

Constraints:
- keep the same `run(chars, features) -> DataFrame(id, eom, w)` interface
- keep rolling/expanding support
- keep deterministic seeds
- use cheap smoke-stage defaults (small trees / moderate estimators / shallow MLP)

## Task 3: Add model registry and public exports

Files:
- Create: `eapctf/ctf/model_registry.py`
- Modify: `eapctf/ctf/__init__.py`

Registry requirements:
- `MODEL_REGISTRY` mapping string names to classes
- `available_model_families()`
- `build_point_baseline(name, **kwargs)`
- `register_model(name, cls)` for future custom injection

## Task 4: Document custom model addition path

Files:
- Create: `docs/model-selection-and-custom-models.md`

Content requirements:
- supported built-in families
- how to add a custom model class
- required methods / constructor expectations
- minimal example: register custom OLS-derived class via `register_model`
- smoke-test checklist

## Task 5: Add server2 smoke runner for all model families

Files:
- Create: `reference/run_model_family_smoke.py`

Runner requirements:
- use `available_model_families()`
- instantiate each model with cheap smoke configs
- run on reduced config via env vars:
  - `EAPCTF_MAX_FEATURES`
  - `EAPCTF_MAX_TEST_MONTHS`
- log progress JSON lines per family
- on failure, capture exception in output and continue to next family
- final output must summarize pass/fail and basic metrics per model family

## Task 6: Run smoke tests on server2 in background

Commands:
- unit tests first
- then background smoke runner on server2 with reduced settings

Recommended reduced config:
- `EAPCTF_MAX_FEATURES=20`
- `EAPCTF_MAX_TEST_MONTHS=12`

Success criteria:
- all model families attempt to run
- failures, if any, are explicit and isolated by family
- user can inspect the log next morning without reading code
