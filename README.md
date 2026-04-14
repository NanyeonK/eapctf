# eapctf

Empirical asset pricing toolkit under rebuild.

Current rebuild scope:
- reset package surface to a minimal, auditable baseline
- remove legacy `conformal.py` naming from the active package surface
- restore point-prediction benchmark path before any uncertainty claims
- encode fixed CTF rules separately from participant choices and research choices
- rebuild uncertainty methods around a joint point+uncertainty interface

Current active module surface:
- `eapctf.ctf.uncertainty`

Core contract split:
- fixed CTF contract: inputs, submission schema, test scope, server metric, compliance rules
- participant choice contract: feature handling, model, tuning, training schedule, signal-to-weight mapping
- research contract: hold IPCA fixed as benchmark anchor, then test joint point+uncertainty models against it

Status:
- bootstrap skeleton with benchmark parity path restored
- unified point+uncertainty architecture defined
- next runnable step is first IPCA joint model implementation

Reference docs:
- `benchmark_contract.yaml`
- `docs/ctf-contract-boundaries.md`
- `docs/rebuild-roadmap.md`
