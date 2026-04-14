# eapctf rebuild roadmap

## Immediate objective
1. Reproduce the archived IPCA benchmark as closely as possible.
2. Check whether its local CTF metric is close to the live leaderboard range.
3. Only then add a light uncertainty overlay.

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
6. Add minimal uncertainty hook only after parity is acceptable
