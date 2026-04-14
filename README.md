# eapctf

Empirical asset pricing toolkit under rebuild.

Current rebuild scope:
- reset package surface to a minimal, auditable baseline
- remove legacy `conformal.py` naming from the active package surface
- rebuild uncertainty methods around explicit interval/residual uncertainty contracts
- restore point-prediction benchmark path before any uncertainty claims

Current active module surface:
- `eapctf.ctf.uncertainty`

Status:
- bootstrap skeleton only
- no backward-compatibility guarantees yet
