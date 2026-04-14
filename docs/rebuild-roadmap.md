# eapctf rebuild roadmap

## Immediate objective
1. Reproduce the archived IPCA benchmark as closely as possible.
2. Check whether its local CTF metric is close to the live leaderboard range.
3. Only then introduce a unified point+uncertainty model path.

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
6. Build a unified point+uncertainty interface for IPCA first

## Unified uncertainty policy
- Reject the old post-hoc two-stage `q_hat` pipeline as the active design.
- Each fitted model must emit point and uncertainty together at predict time.
- Benchmark output and uncertainty-augmented output remain separate artifacts.
- First unified target is IPCA, because IPCA parity is already established.
- Candidate uncertainty objects on top of IPCA:
  - exposure instability
  - factor reconstruction error
  - rolling residual volatility
