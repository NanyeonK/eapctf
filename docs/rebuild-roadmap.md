# eapctf rebuild roadmap

## Immediate objective
1. Reproduce the archived IPCA benchmark as closely as possible.
2. Check whether its local CTF metric is close to the live leaderboard range.
3. Lock the CTF fixed contract before changing any modeling layer.
4. Use IPCA as the first audited anchor, not as the permanent sole architecture.
5. Move to a model-agnostic joint point+uncertainty interface.

## Contract boundaries
The rebuild must keep three layers separate.

### Fixed CTF contract
Frozen and not a research choice:
- inputs: `chars`, `features`, `daily_ret`
- submission: `main(chars, features, daily_ret) -> DataFrame(id, eom, w)`
- test scope: `ctff_test`
- server metric: 10% vol-targeted Sharpe using next-month returns from `daily_ret`
- compliance: no look-ahead, no NaN, unique `(id, eom)`, deterministic output

### Participant choice contract
Allowed strategy variation inside the fixed shell:
- feature subset and preprocessing
- model class and hyperparameters
- rolling/expanding training design
- refit schedule
- signal-to-weight mapping
- uncertainty use inside weight construction

### Research contract for this project
What we are actually testing:
- IPCA is the first audited anchor because parity is already established
- the package architecture must become model-family agnostic
- every fitted object must emit point and uncertainty jointly
- reject the old detached post-hoc `q_hat` pipeline as the active rebuild design
- every uncertainty experiment must report delta versus the same model's point baseline and versus the IPCA anchor

## Archived references to reuse
- Archived repo root:
  `~/project_archive/eapctf_local_reset_20260414_165453/eapctf_before_rebuild/`
- Archived benchmark script:
  `reference/benchmark-ipca-pf.py`
- Archived weight artifact:
  `data/ctf/ipca_weights.parquet`
- Archived leaderboard utility:
  `eapctf/ctf/leaderboard.py`

## Rebuild order
1. Restore leaderboard utility
2. Restore or rewrite IPCA reference benchmark runner
3. Restore metric computation needed to compare local run vs leaderboard
4. Run archived IPCA benchmark path on current data
5. Compare local metric vs archived metric vs live leaderboard
6. Generalize the joint interface so multiple point-model families can plug in
7. Keep IPCA as audited anchor while adding model-specific point baselines
8. Add uncertainty objects model by model and report both deltas

## Unified uncertainty policy
- Reject the old post-hoc two-stage `q_hat` pipeline as the active design.
- Each fitted model must emit point and uncertainty together at predict time.
- Benchmark output and uncertainty-augmented output remain separate artifacts.
- IPCA is first because IPCA parity is already established, not because it is the only allowed baseline.
- Candidate uncertainty objects on top of IPCA first:
  - exposure instability
  - factor reconstruction error
  - rolling residual volatility
- After IPCA, expand the same contract to OLS and non-linear ML families.
- Contract reference:
  `docs/ctf-contract-boundaries.md`
