# eapctf rebuild roadmap

## Immediate objective
1. Reproduce the archived IPCA benchmark as closely as possible.
2. Check whether its local CTF metric is close to the live leaderboard range.
3. Lock the CTF fixed contract before changing any modeling layer.
4. Only then introduce a unified point+uncertainty model path.

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
- IPCA stays fixed as benchmark anchor
- every fitted object must emit point and uncertainty jointly
- reject the old detached post-hoc `q_hat` pipeline as the active rebuild design
- every uncertainty experiment must report delta vs the fixed benchmark path

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
6. Implement first unified IPCA model object with `predict_with_uncertainty`
7. Run one conservative uncertainty object and report delta vs benchmark

## Unified uncertainty policy
- Reject the old post-hoc two-stage `q_hat` pipeline as the active design.
- Each fitted model must emit point and uncertainty together at predict time.
- Benchmark output and uncertainty-augmented output remain separate artifacts.
- First unified target is IPCA, because IPCA parity is already established.
- Candidate uncertainty objects on top of IPCA:
  - exposure instability
  - factor reconstruction error
  - rolling residual volatility
- Contract reference:
  `docs/ctf-contract-boundaries.md`
