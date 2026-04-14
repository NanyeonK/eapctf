#!/usr/bin/env python3
"""Reference IPCA benchmark placeholder for the rebuild.

Purpose:
- recover the archived IPCA path that previously tracked the leaderboard closely
- verify local metric parity before any uncertainty extension

Archived references:
- ~/project_archive/eapctf_local_reset_20260414_165453/eapctf_before_rebuild/reference/benchmark-ipca-pf.py
- ~/project_archive/eapctf_local_reset_20260414_165453/eapctf_before_rebuild/data/ctf/ipca_weights.parquet

Rebuild note:
This file is intentionally a bootstrap placeholder. Port the old logic only
after wiring the current package metric path and benchmark contract.
"""

from __future__ import annotations

from pathlib import Path

ARCHIVE_ROOT = Path.home() / "project_archive" / "eapctf_local_reset_20260414_165453" / "eapctf_before_rebuild"
ARCHIVED_BENCHMARK = ARCHIVE_ROOT / "reference" / "benchmark-ipca-pf.py"
ARCHIVED_WEIGHTS = ARCHIVE_ROOT / "data" / "ctf" / "ipca_weights.parquet"


def main() -> None:
    print("IPCA rebuild benchmark placeholder")
    print(f"archived benchmark script: {ARCHIVED_BENCHMARK}")
    print(f"archived ipca weights: {ARCHIVED_WEIGHTS}")
    print("Next step: port benchmark logic and metric comparison into rebuilt package.")


if __name__ == "__main__":
    main()
