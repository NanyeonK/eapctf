"""CTF subpackage bootstrap.

Legacy active surface used `conformal.py` as the uncertainty module name.
That name is intentionally retired from the rebuilt active package surface.
Use `eapctf.ctf.uncertainty` instead.
"""

from eapctf.ctf.uncertainty import UncertaintyConfig

__all__ = ["UncertaintyConfig"]
