"""CTF leaderboard scraper and rank estimator.

Restored first because the rebuild objective is to compare archived IPCA
performance against the live leaderboard before adding uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class LeaderboardEntry:
    rank: int
    model_name: str
    sharpe: float
    annual_return: float | None = None
    volatility: float | None = None


@dataclass
class LeaderboardResult:
    entries: list[LeaderboardEntry] = field(default_factory=list)
    source: str = "unknown"

    @property
    def top_sharpe(self) -> float | None:
        return max((e.sharpe for e in self.entries), default=None)

    def estimate_rank(self, sharpe: float) -> int:
        if not self.entries:
            return 1
        return sum(1 for e in self.entries if e.sharpe > sharpe) + 1


def fetch_leaderboard(timeout: int = 10) -> LeaderboardResult:
    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required to fetch leaderboard") from exc

    url = "https://jkpfactors.com/ctf"
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "eapctf-rebuild/0.7"})
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    if not tables:
        return LeaderboardResult(entries=[], source="unavailable")

    lb_table = None
    for tbl in tables:
        cols_lower = [str(c).lower() for c in tbl.columns]
        if any("sharpe" in c for c in cols_lower):
            lb_table = tbl
            break
    if lb_table is None:
        return LeaderboardResult(entries=[], source="unavailable")

    cols_lower = [str(c).lower() for c in lb_table.columns]

    def find_col(keys: list[str]):
        for key in keys:
            for orig, low in zip(lb_table.columns, cols_lower, strict=False):
                if key in low:
                    return orig
        return None

    sharpe_col = find_col(["sharpe"])
    name_col = find_col(["model", "name", "method", "algorithm"])
    ret_col = find_col(["return", "ret"])
    vol_col = find_col(["vol", "volatility"])
    if sharpe_col is None:
        return LeaderboardResult(entries=[], source="unavailable")

    entries: list[LeaderboardEntry] = []
    for rank_idx, (_, row) in enumerate(lb_table.iterrows(), start=1):
        try:
            sharpe = float(row[sharpe_col])
        except Exception:
            continue
        name = str(row[name_col]) if name_col else f"Model #{rank_idx}"
        ann_ret = None
        vol = None
        if ret_col is not None:
            try:
                ann_ret = float(row[ret_col])
            except Exception:
                pass
        if vol_col is not None:
            try:
                vol = float(row[vol_col])
            except Exception:
                pass
        entries.append(LeaderboardEntry(rank_idx, name, sharpe, ann_ret, vol))

    entries.sort(key=lambda e: e.sharpe, reverse=True)
    for idx, entry in enumerate(entries, start=1):
        entry.rank = idx
    return LeaderboardResult(entries=entries, source="scraped")
