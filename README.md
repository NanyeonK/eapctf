# eapctf

Empirical asset pricing toolkit under rebuild.

Current rebuild scope:
- reset package surface to a minimal, auditable baseline
- remove legacy `conformal.py` naming from the active package surface
- restore point-prediction benchmark path before any uncertainty claims
- encode fixed CTF rules separately from participant choices and research choices
- rebuild uncertainty methods around a joint point+uncertainty interface
- generalize that interface so multiple model families can use it

Core contract split:
- fixed CTF contract: inputs, submission schema, test scope, server metric, compliance rules
- participant choice contract: feature handling, model, tuning, training schedule, signal-to-weight mapping
- research contract:
  - IPCA is the first audited rebuild anchor
  - not the permanent sole model baseline
  - the permanent architecture target is model-family agnostic joint prediction + uncertainty

Status:
- bootstrap skeleton with benchmark parity path restored
- unified point+uncertainty architecture defined
- first IPCA-native uncertainty objects implemented as anchor-stage prototypes
- next architectural step is generalizing beyond IPCA-specific wrappers

Reference docs:
- `benchmark_contract.yaml`
- `docs/ctf-contract-boundaries.md`
- `docs/rebuild-roadmap.md`
