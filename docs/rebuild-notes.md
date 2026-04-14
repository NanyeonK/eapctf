# eapctf rebuild notes

The old active uncertainty module name `conformal.py` is retired.
The rebuilt package starts from `eapctf/ctf/uncertainty.py`.

Migration rule:
- do not create a new active `eapctf/ctf/conformal.py`
- if old logic is recovered from backups, port only the needed pieces into
  the new uncertainty stack after review
