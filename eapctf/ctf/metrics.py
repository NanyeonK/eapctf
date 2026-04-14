"""CTF performance metrics for submitted portfolio weights.

Computes ex-ante performance metrics from (weights_df, daily_ret_df) following
the HJKP 2025 Common Task Framework evaluation methodology:

- Monthly portfolio returns via weight × daily return aggregation
- Ex-ante Sharpe ratio: annualized (mean / std × sqrt(12))
- Annual return: geometric mean of monthly returns
- Annualized volatility: std(monthly) × sqrt(12)
- Maximum drawdown: peak-to-trough over the evaluation window
- Benchmark comparison: equal-weight (1/N) vs submitted model

Usage::

    result = compute_metrics(weights_df, daily_ret_df)
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MetricsResult:
    """Performance metrics for a submitted portfolio.

    Attributes
    ----------
    sharpe : float
        Ex-ante annualized Sharpe ratio (monthly mean / std × sqrt(12)).
    annual_return : float
        Annualized geometric mean return (compounded monthly).
    volatility : float
        Annualized return volatility (monthly std × sqrt(12)).
    max_drawdown : float
        Maximum peak-to-trough drawdown (negative or zero).
    n_months : int
        Number of monthly return observations.
    monthly_returns : pd.Series
        Monthly portfolio excess returns (index: eom date).
    benchmark_sharpe : float | None
        Sharpe of the equal-weight (1/N) benchmark, if computed.
    benchmark_annual_return : float | None
        Annual return of the 1/N benchmark.
    """

    sharpe: float
    annual_return: float
    volatility: float
    max_drawdown: float
    n_months: int
    monthly_returns: pd.Series
    benchmark_sharpe: float | None = None
    benchmark_annual_return: float | None = None

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "CTF PERFORMANCE METRICS",
            "=" * 50,
            f"  Sharpe ratio (annualized) : {self.sharpe:+.3f}",
            f"  Annual return             : {self.annual_return:+.2%}",
            f"  Annualized volatility     : {self.volatility:.2%}",
            f"  Max drawdown              : {self.max_drawdown:.2%}",
            f"  Months evaluated          : {self.n_months}",
        ]
        if self.benchmark_sharpe is not None:
            lines += [
                "",
                "  --- 1/N Benchmark ---",
                f"  Sharpe (1/N)              : {self.benchmark_sharpe:+.3f}",
                f"  Annual return (1/N)       : {self.benchmark_annual_return:+.2%}",
            ]
        lines.append("=" * 50)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------


def _monthly_portfolio_returns(
    weights_df: pd.DataFrame,
    daily_ret_df: pd.DataFrame,
    id_col: str = "id",
    eom_col: str = "eom",
    w_col: str = "w",
    date_col: str = "date",
    ret_col: str = "ret_exc",
) -> pd.Series:
    """Aggregate daily returns into monthly portfolio returns.

    For each portfolio formation date (eom), sum daily excess returns
    over the following month, weighted by the portfolio weights at eom.

    The CTF convention: weights at ``eom`` earn returns from ``eom`` to
    ``eom + 1 month`` (i.e., returns with dates in the following calendar
    month).

    Parameters
    ----------
    weights_df : pd.DataFrame
        Columns: id, eom, w.  eom is the portfolio formation date.
    daily_ret_df : pd.DataFrame
        Columns: id, date, ret_exc.  Daily excess returns.
    id_col, eom_col, w_col : str
        Column names in weights_df.
    date_col, ret_col : str
        Column names in daily_ret_df.

    Returns
    -------
    pd.Series
        Monthly portfolio returns indexed by eom date.
    """
    # Normalise date types
    weights = weights_df.copy()
    daily = daily_ret_df.copy()

    weights[eom_col] = pd.to_datetime(weights[eom_col])
    daily[date_col] = pd.to_datetime(daily[date_col])

    # Map each eom → the next calendar month (return period)
    weights["_ret_month"] = weights[eom_col].dt.to_period("M") + 1

    # Tag daily returns with their year-month period
    daily["_ret_month"] = daily[date_col].dt.to_period("M")

    # Merge weights with daily returns on (id, return_month)
    merged = weights.merge(
        daily[[id_col, "_ret_month", ret_col]],
        on=[id_col, "_ret_month"],
        how="inner",
    )

    if merged.empty:
        return pd.Series(dtype=float)

    # Monthly portfolio return = sum_i(w_i * sum_t(r_it)) for the return month.
    # This is the simple-return approximation; adequate for daily r_it ~ 0.1%.
    monthly = (
        merged
        .groupby(eom_col)
        .apply(lambda g: (g[w_col] * g[ret_col]).sum(), include_groups=False)  # type: ignore[call-overload]
    )
    monthly.index = pd.to_datetime(monthly.index)
    monthly.name = "portfolio_ret"
    return monthly.sort_index()  # type: ignore[no-any-return]


def _sharpe(monthly_returns: pd.Series) -> float:
    """Annualized Sharpe ratio: mean / std × sqrt(12)."""
    if len(monthly_returns) < 2:
        return float("nan")
    mu = float(monthly_returns.mean())
    sigma = float(monthly_returns.std(ddof=1))
    if sigma == 0:
        return float("nan")
    return mu / sigma * np.sqrt(12)  # type: ignore[no-any-return]


def _annual_return(monthly_returns: pd.Series) -> float:
    """Annualized geometric mean return from monthly returns."""
    if len(monthly_returns) == 0:
        return float("nan")
    gross = 1.0 + monthly_returns.values  # type: ignore[operator]
    n = len(gross)  # type: ignore[arg-type]
    # Log-sum is numerically stable vs np.prod (avoids underflow for long series)
    if np.any(gross <= 0):
        return float("nan")
    log_compound = float(np.sum(np.log(gross)))
    return float(np.exp(log_compound * 12.0 / n) - 1.0)


def _volatility(monthly_returns: pd.Series) -> float:
    """Annualized volatility: std(monthly) × sqrt(12)."""
    if len(monthly_returns) < 2:
        return float("nan")
    return float(monthly_returns.std(ddof=1)) * np.sqrt(12)  # type: ignore[no-any-return]


def _max_drawdown(monthly_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative or zero)."""
    if len(monthly_returns) == 0:
        return float("nan")
    cumulative = (1.0 + monthly_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_metrics(
    weights_df: pd.DataFrame,
    daily_ret_df: pd.DataFrame,
    id_col: str = "id",
    eom_col: str = "eom",
    w_col: str = "w",
    date_col: str = "date",
    ret_col: str = "ret_exc",
    compute_benchmark: bool = True,
    vol_target: float | None = 0.10,
) -> MetricsResult:
    """Compute ex-ante performance metrics for CTF submission weights.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Portfolio weights with columns ``id``, ``eom``, ``w``.
    daily_ret_df : pd.DataFrame
        Daily excess returns with columns ``id``, ``date``, ``ret_exc``.
    id_col, eom_col, w_col : str
        Column names in weights_df.
    date_col, ret_col : str
        Column names in daily_ret_df.
    compute_benchmark : bool
        If True, also compute equal-weight (1/N) benchmark metrics.
    vol_target : float | None
        If set, scale monthly returns so that annualized volatility equals
        this target (default 0.10 = 10%, matching CTF server evaluation).
        Sharpe ratio is invariant to this scaling; annual_return and
        max_drawdown are reported in vol-targeted units.  Set None to
        disable scaling and use raw portfolio returns.

    Returns
    -------
    MetricsResult
        Sharpe, annual return, vol, MDD, and optional benchmark stats.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    for col in (id_col, eom_col, w_col):
        if col not in weights_df.columns:
            raise ValueError(f"weights_df missing column '{col}'")
    for col in (id_col, date_col, ret_col):
        if col not in daily_ret_df.columns:
            raise ValueError(f"daily_ret_df missing column '{col}'")

    monthly = _monthly_portfolio_returns(
        weights_df, daily_ret_df,
        id_col=id_col, eom_col=eom_col, w_col=w_col,
        date_col=date_col, ret_col=ret_col,
    )

    # Vol-target scaling: scale monthly returns so annualized vol = vol_target.
    # Sharpe is invariant; annual_return and max_drawdown reflect scaled units.
    if vol_target is not None and len(monthly) >= 2:
        realized_vol = float(monthly.std(ddof=1)) * np.sqrt(12)
        if realized_vol > 0:
            monthly = monthly * (vol_target / realized_vol)

    result = MetricsResult(
        sharpe=_sharpe(monthly),
        annual_return=_annual_return(monthly),
        volatility=_volatility(monthly),
        max_drawdown=_max_drawdown(monthly),
        n_months=len(monthly),
        monthly_returns=monthly,
    )

    if compute_benchmark and not daily_ret_df.empty:
        # Equal-weight benchmark: mean daily excess return per calendar month
        bm_daily = daily_ret_df[[id_col, date_col, ret_col]].copy()
        bm_daily[date_col] = pd.to_datetime(bm_daily[date_col])
        bm_daily["_ret_month"] = bm_daily[date_col].dt.to_period("M")

        ew_monthly = (
            bm_daily.groupby("_ret_month")[ret_col].mean()
        )
        ew_monthly.index = ew_monthly.index.to_timestamp()  # type: ignore[attr-defined]
        ew_monthly = ew_monthly.sort_index()

        # Restrict benchmark to the same return months as the portfolio.
        # Portfolio monthly.index holds eom formation dates; the corresponding
        # return months are one period later.
        if len(monthly) > 0:
            port_ret_months = pd.to_datetime(
                (monthly.index.to_period("M") + 1).to_timestamp()  # type: ignore[attr-defined]
            )
            ew_monthly = ew_monthly[
                (ew_monthly.index >= port_ret_months.min()) &  # type: ignore[operator]
                (ew_monthly.index <= port_ret_months.max())  # type: ignore[operator]
            ]

        result.benchmark_sharpe = _sharpe(ew_monthly)
        result.benchmark_annual_return = _annual_return(ew_monthly)

    return result
