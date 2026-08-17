"""Measurement + inference for Study 960 — The Unstaked ETF.

The question. A US spot-Ethereum ETF holds coins and (over most of this sample) does
not stake them, so its holder forgoes the consensus reward an on-chain staker earns.
How much does the wrapper cost, and what share of that cost is the forgone yield
rather than the sponsor fee?

The three measurements, in the order they must be read:

1. **Fund vs coin.** ``tracking_difference(fund, ETH-USD)`` — the realised annualised
   gap between a fund's close and the coin's close. This is the number the question
   asks for, and it is the one the tape answers *badly*: the coin prints 24/7 and is
   stamped hours after the 16:00 fund close, so the daily difference carries a large,
   transient, strongly mean-reverting timing error on a ~70%-vol asset.

2. **Fund vs fund.** ``tracking_difference(ETHE, ETHA)`` — the same estimator applied
   to two wrappers on the *same* coin, stamped at the *same* close. The timing error
   cancels exactly, and what is left is a clean read on a documented cost difference.
   This is the study's calibration instrument: it tells you what size of drag this
   tape *can* resolve, which is the honest answer to the headline question.

3. **The ledger.** ``wrapper_ledger`` adds the two ASSUMED constants — the sponsor fee
   and the forgone staking yield — on top of the measured tracking difference and
   reports the split. Crucially, ETH-USD is itself the price of an **unstaked** coin,
   so the forgone yield *cannot* appear in measurement 1 at all: it is a bookkeeping
   add-on relative to a self-staking holder, never a tape observation. Saying so
   plainly is most of this study.

Execution lag. There is exactly one, and only measurement 4 uses it: the long/short
fee-spread capture in ``long_short_capture`` forms its (constant) target weights from
information through day ``t`` and holds them from ``t+1``. Measurements 1-3 are
accounting on contemporaneous closes and involve no trading decision, so no lag
applies to them.

Costs. One-way x NAV on every trade; the short leg pays an explicit annual borrow,
swept, because a 2%-a-year spread does not survive a hard-to-borrow line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives (shared desk mirror)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0.

    **Read the direction before you read the number.** On these series the daily
    difference is *negatively* autocorrelated (the timing and premium/discount errors
    reverse the next session), so the HAC variance is *smaller* than the i.i.d. one and
    the HAC t is *larger* in absolute value. We are therefore **not** taking the most
    conservative reading available: the plain i.i.d. daily t is the widest of the three
    rulers here, and on the ETHE-ETHA spread it comes in at only -1.32.

    We do not use it, and the reason is stated rather than hidden. A difference of two
    closes on the same asset with a one-session reversal is close to an MA(1) with
    rho1 ~ -0.5; the variance of its *sum* over k days grows far more slowly than k, so
    the i.i.d. daily t assumes away a real, mechanical feature of the data and is
    over-wide, not "safe". The rulers this study reads are the two that do not depend
    on that assumption: a **circular block bootstrap** (block sweep published, see
    :func:`resolution_sweep`) and a **non-overlapping monthly** t on 24 calendar-month
    sums. Both are quoted next to the i.i.d. and HAC readings everywhere, so a reader
    who disagrees with the choice can see exactly what it bought.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def _hac_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised stats for a daily simple-return series (pass an excess series for
    the excess Sharpe)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu, std = r.mean(), r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = (float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
            if years > 0 and wealth.iloc[-1] > 0 else float("nan"))
    return {
        "n_days": int(n),
        "cagr": cagr,
        "sharpe": sharpe,
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy(), lags=_hac_lags(n)),
    }


# --------------------------------------------------------------------------- #
# 1-2. Tracking difference
# --------------------------------------------------------------------------- #
def log_diff(fund: pd.Series, bench: pd.Series) -> pd.Series:
    """Daily log-return difference ``fund - bench`` on the common calendar.

    Log returns, not simple ones, because a tracking difference is a *compounding*
    drag: the sum of daily log differences equals the log of the terminal wealth
    ratio, so the estimator telescopes cleanly to the endpoint result.
    """
    common = fund.index.intersection(bench.index)
    f = np.log(fund.reindex(common).sort_index().astype(float))
    b = np.log(bench.reindex(common).sort_index().astype(float))
    return (f.diff() - b.diff()).dropna().rename("log_diff")


def tracking_difference(fund: pd.Series, bench: pd.Series) -> dict:
    """Annualised realised tracking difference of ``fund`` against ``bench``.

    Reports the number three ways, because on this tape they disagree in *precision*
    rather than in level and the reader must see that:

    - ``ann_td_pct`` — mean daily log difference x 252, in % per year. This is the
      compounding-correct one: its sum *is* the log terminal wealth ratio, so it is what
      a holder who switched wrappers actually kept.
    - ``ann_td_geo_pct`` — the difference of the two **annualised simple** returns. It is
      reported because it is what most fund factsheets show, but it is level-dependent:
      on an asset that halved, differencing two annualised simple returns shrinks the
      gap (ETHE-ETHA reads -1.00%/yr this way against -1.64%/yr on the log measure, for
      an identical +3.1% of terminal wealth over 1.92 years). Never quote it as "the
      holder's benefit"; quote ``ann_td_pct`` for that.
    - ``iid_t`` / ``hac_t`` / ``monthly_t`` — the same mean under three variance
      assumptions. A wide gap between ``iid_t`` and ``hac_t`` is the signature of the
      mean-reverting observation error, not of a stronger effect.
    """
    d = log_diff(fund, bench)
    n = len(d)
    if n < 30:
        return {"n_days": n, "ann_td_pct": float("nan")}
    common = d.index
    years = n / TRADING_DAYS
    f0, f1 = float(fund.loc[common[0]]), float(fund.loc[common[-1]])
    b0, b1 = float(bench.loc[common[0]]), float(bench.loc[common[-1]])
    geo = ((f1 / f0) ** (1.0 / years) - (b1 / b0) ** (1.0 / years)) * 100.0

    monthly = d.resample("ME").sum()
    m_n = len(monthly)
    m_sd = float(monthly.std(ddof=1))
    m_t = float(monthly.mean() / (m_sd / np.sqrt(m_n))) if m_sd > 0 and m_n > 2 else float("nan")

    return {
        "n_days": int(n),
        "years": float(years),
        "start": str(common[0].date()),
        "end": str(common[-1].date()),
        "ann_td_pct": float(d.mean() * TRADING_DAYS * 100.0),
        "ann_td_geo_pct": float(geo),
        "daily_sd_bps": float(d.std(ddof=1) * 1e4),
        "iid_t": one_sample_t(d.to_numpy()),
        "hac_t": newey_west_t(d.to_numpy(), lags=_hac_lags(n)),
        "monthly_t": m_t,
        "monthly_n": int(m_n),
        "autocorr1": float(d.autocorr(1)),
        "diff": d,
    }


def bootstrap_td_ci(
    diff: pd.Series,
    n_boot: int = 4000,
    block: int = 21,
    seed: int = 960,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for an annualised tracking difference (% / yr).

    Blocks of ``block`` consecutive days keep the day-to-day reversal inside a block,
    so the interval respects the mean-reverting observation error instead of assuming
    it away. This is the interval the verdict is read from — but the width is **not**
    block-free, and on the fund-vs-coin series it is strongly block-dependent, because
    a mean-reverting error telescopes further the longer the block. Never quote a single
    width off this function without :func:`resolution_sweep` beside it.
    """
    r = np.asarray(pd.Series(diff).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_obs": int(n)}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean() * TRADING_DAYS * 100.0
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(r.mean() * TRADING_DAYS * 100.0),
        "ci_low": float(lo), "ci_high": float(hi),
        "half_width": float((hi - lo) / 2.0),
        "frac_negative": float((boots < 0).mean()),
        "n_obs": int(n), "n_boot": int(n_boot), "block": int(block),
    }


def resolvable_drag(diff: pd.Series, block: int = 21, seed: int = 960) -> float:
    """The smallest annual drag this tape could distinguish from zero at |t| = 2.

    Half the bootstrap interval width, in % per year — the study's resolution limit at
    one block length. If the forgone staking yield is smaller than this, the tape simply
    cannot see it, whatever the point estimate happens to be. **Always publish
    :func:`resolution_sweep` alongside it**: on a mean-reverting difference this number
    is a function of the block, and quoting one block is quoting a choice.
    """
    ci = bootstrap_td_ci(diff, block=block, seed=seed)
    return float(ci["half_width"])


def monthly_halfwidth(diff: pd.Series) -> float:
    """Resolution limit from the blunt, assumption-free ruler: non-overlapping months.

    Sum the daily differences inside each calendar month, treat those sums as
    independent observations (they are non-overlapping by construction — no HAC, no
    block choice, no kernel), and convert 1.96 standard errors back to % per year. This
    is the *narrowest* honest limit on this tape, which is why it is published next to
    the widest one rather than instead of it.
    """
    d = pd.Series(diff).dropna()
    if len(d) < 60:
        return float("nan")
    m = d.resample("ME").sum()
    if len(m) < 3:
        return float("nan")
    days_per_month = len(d) / len(m)
    se_ann = float(m.std(ddof=1) / np.sqrt(len(m))) * (TRADING_DAYS / days_per_month) * 100.0
    return float(1.96 * se_ann)


def resolution_sweep(diff: pd.Series, blocks=(5, 10, 21, 42, 63), seed: int = 960) -> dict:
    """How much of the resolution limit is the *ruler* rather than the tape.

    The headline "this tape resolves +/-X %/yr" is a bootstrap half-width, and on a
    negatively autocorrelated difference the half-width falls as the block lengthens
    (the reversal telescopes inside longer blocks). Sweeping the block, and adding the
    non-overlapping monthly limit, turns one convenient number into a range the reader
    can argue with. Returns ``{"blocks": {b: halfwidth}, "monthly": halfwidth,
    "min": ..., "max": ...}``, all in % per year.
    """
    per_block = {int(b): float(bootstrap_td_ci(diff, block=int(b), seed=seed)["half_width"])
                 for b in blocks}
    monthly = monthly_halfwidth(diff)
    widths = [w for w in list(per_block.values()) + [monthly] if np.isfinite(w)]
    return {
        "blocks": per_block,
        "monthly": float(monthly),
        "min": float(min(widths)) if widths else float("nan"),
        "max": float(max(widths)) if widths else float("nan"),
    }


# --------------------------------------------------------------------------- #
# 3. The wrapper-cost ledger (measured tape + declared assumptions)
# --------------------------------------------------------------------------- #
def wrapper_ledger(
    td_vs_coin_pct: float,
    expense_ratio: float,
    staking_yield: float,
) -> dict:
    """Split the total cost of the wrapper into measured and assumed components.

    Parameters
    ----------
    td_vs_coin_pct:
        The MEASURED annualised tracking difference of the fund against ETH-USD, in %
        per year (negative = the fund lagged the coin).
    expense_ratio, staking_yield:
        ASSUMPTIONS, both in decimal per year. Neither is observed on this tape: the
        first is the documented sponsor fee, the second the consensus reward a staked
        coin would have earned. They are swept, never asserted.

    The arithmetic that matters: ETH-USD is the price of an **unstaked** coin, so the
    forgone staking yield is *absent by construction* from ``td_vs_coin_pct``. The
    holder's total shortfall against a **self-staking** coin holder is therefore the
    measured tracking difference *minus* the staking yield, and the staking share of
    that shortfall is what the study is asked to quantify.
    """
    fee_pct = expense_ratio * 100.0
    stake_pct = staking_yield * 100.0
    total = td_vs_coin_pct - stake_pct          # vs a self-staking coin holder
    vs_unstaked = td_vs_coin_pct                # vs an unstaked coin holder (the tape)
    denom = abs(fee_pct) + abs(stake_pct)
    return {
        "td_vs_coin_pct": float(td_vs_coin_pct),
        "fee_pct_assumed": float(fee_pct),
        "staking_pct_assumed": float(stake_pct),
        "total_vs_staked_holder_pct": float(total),
        "total_vs_unstaked_holder_pct": float(vs_unstaked),
        "staking_share_of_declared_cost": float(stake_pct / denom) if denom > 0 else float("nan"),
        "unexplained_pct": float(td_vs_coin_pct + fee_pct),  # measured minus the fee we expected
    }


def staking_sweep(
    td_vs_coin_pct: float,
    expense_ratio: float,
    yields=(0.020, 0.025, 0.030, 0.035, 0.045),
) -> list[dict]:
    """Sweep the ASSUMED staking yield through the ledger.

    The staking yield is the single number in this study that comes from outside the
    tape, so it gets a sweep rather than a point value. Nothing else in the ledger
    moves with it — which is itself the finding.
    """
    return [wrapper_ledger(td_vs_coin_pct, expense_ratio, y) for y in yields]


def fee_sweep(
    td_vs_coin_pct: float,
    staking_yield: float,
    fees=(0.0012, 0.0025),
) -> list[dict]:
    """Sweep the ASSUMED effective sponsor fee (the early waiver makes it a range)."""
    return [wrapper_ledger(td_vs_coin_pct, f, staking_yield) for f in fees]


# --------------------------------------------------------------------------- #
# 4. Is the measured spread bankable? (the only leg that trades)
# --------------------------------------------------------------------------- #
def long_short_capture(
    cheap: pd.Series,
    dear: pd.Series,
    cash: pd.Series,
    borrow_bps_ann: float = 100.0,
    cost_bps: float = 10.0,
    holding_years: float = 1.0,
) -> dict:
    """Harvest the measured fee spread: long the cheap wrapper, short the dear one.

    Dollar-neutral, one unit a side, held for the whole window. Weights are constant
    and are formed from information through day ``t`` (the two published sponsor fees)
    and held from ``t+1`` — the study's single execution lag. Charges:

    - ``cost_bps`` one-way x NAV on each of the two legs, twice (in and out);
    - ``borrow_bps_ann`` per year x NAV on the short leg. This is the number that
      decides the verdict, so it is swept: a Grayscale-style line is not a
      general-collateral borrow, and a 2%-a-year spread has very little room.

    ``cash`` is used **only** to align the calendar: the book is dollar-neutral, so the
    short proceeds fund the long leg and there is no net cash balance to accrue. No cash
    return is credited to this trade — do not read the ``cash`` argument as a rebate.

    Reported both ways round on inference: ``hac_t_gross`` is the flattering direction
    here (negative autocorrelation shrinks the HAC variance), so ``monthly_t_gross`` —
    24 non-overlapping calendar-month sums, no kernel, no block — is the reading to
    quote, with ``iid_t_gross`` shown as the over-wide bound.
    """
    common = cheap.index.intersection(dear.index).intersection(cash.index)
    c = cheap.reindex(common).sort_index()
    d = dear.reindex(common).sort_index()
    rc = c.pct_change()
    rd = d.pct_change()
    lag = 1  # the single execution lag: weights known at t, applied from t+1
    # Drop the pct_change warm-up bar FIRST, then apply the lag, so the lag is a real
    # forfeited session rather than an accounting coincidence with the NaN row.
    gross = (rc - rd).dropna().iloc[lag:]
    n = len(gross)
    borrow_daily = borrow_bps_ann * 1e-4 / TRADING_DAYS
    net = gross - borrow_daily
    roundtrip = 4.0 * cost_bps * 1e-4  # two legs, in and out
    ann_gross = float(gross.mean() * TRADING_DAYS * 100.0)
    ann_net = float(net.mean() * TRADING_DAYS * 100.0) - roundtrip * 100.0 / max(holding_years, 1e-9)
    m = gross.resample("ME").sum()
    m_t = (float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m))))
           if len(m) > 2 and float(m.std(ddof=1)) > 0 else float("nan"))
    return {
        "n_days": int(n),
        "ann_gross_pct": ann_gross,
        "ann_net_pct": ann_net,
        "borrow_bps_ann": float(borrow_bps_ann),
        "cost_bps": float(cost_bps),
        "hac_t_gross": newey_west_t(gross.to_numpy(), lags=_hac_lags(n)),
        "iid_t_gross": one_sample_t(gross.to_numpy()),
        "monthly_t_gross": m_t,
        "monthly_n": int(len(m)),
        "months_positive": int((m > 0).sum()),
        "vol_ann_pct": float(gross.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100.0),
        "max_drawdown": max_drawdown(net),
    }


def borrow_sweep(cheap: pd.Series, dear: pd.Series, cash: pd.Series,
                 borrows=(0.0, 50.0, 100.0, 200.0, 400.0, 800.0),
                 cost_bps: float = 10.0) -> list[dict]:
    """Sweep the short leg's annual borrow cost through the capture trade."""
    return [long_short_capture(cheap, dear, cash, borrow_bps_ann=b, cost_bps=cost_bps)
            for b in borrows]


# --------------------------------------------------------------------------- #
# The excess-of-cash race (fund vs coin, both minus the cash leg)
# --------------------------------------------------------------------------- #
def excess_sharpe_race(fund: pd.Series, coin: pd.Series, cash: pd.Series,
                       cost_bps: float = 0.0) -> dict:
    """Race the wrapper against the coin, both **excess of cash** (minus BIL).

    Included for completeness and read with a straight face: the two arms hold the
    *same* asset, so the race can only ever reveal the drag, and a ~70%-vol asset over
    ~2 years cannot resolve a 3%-a-year drag in Sharpe space. ``cost_bps`` charges a
    single one-way entry on the fund leg (the coin leg is assumed free, which flatters
    the coin — the honest direction here).
    """
    common = fund.index.intersection(coin.index).intersection(cash.index)
    f = fund.reindex(common).sort_index().pct_change()
    c = coin.reindex(common).sort_index().pct_change()
    b = cash.reindex(common).sort_index().pct_change()
    entry = cost_bps * 1e-4
    f = f.copy()
    if len(f) > 1:
        f.iloc[1] = f.iloc[1] - entry
    e_fund = (f - b).dropna().rename("e_fund")
    e_coin = (c - b).dropna().rename("e_coin")
    s_f, s_c = summary(e_fund), summary(e_coin)
    diff = (e_fund - e_coin).dropna()
    return {
        "fund": s_f, "coin": s_c,
        "excess_sharpe_adv": s_f["sharpe"] - s_c["sharpe"],
        "t_diff": newey_west_t(diff.to_numpy(), lags=_hac_lags(len(diff))),
        "iid_t_diff": one_sample_t(diff.to_numpy()),
        "e_fund": e_fund, "e_coin": e_coin,
    }


# --------------------------------------------------------------------------- #
# Era cut
# --------------------------------------------------------------------------- #
def era_cut(fund: pd.Series, bench: pd.Series, split: str) -> dict:
    """Re-measure the tracking difference either side of ``split``.

    A cost that is a *fee* should be flat across eras. A cost that is a forgone
    *staking* yield should shrink the moment a fund starts staking — so an era cut is
    the only tape-side handle this study has on the staking hypothesis, and it is a
    weak one: with ~250 sessions a side the interval is wider still.
    """
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        f, b = fund.loc[sl], bench.loc[sl]
        if len(f) < 60 or len(b) < 60:
            out[tag] = None
            continue
        td = tracking_difference(f, b)
        ci = bootstrap_td_ci(td["diff"])
        out[tag] = {
            "n_days": td["n_days"], "start": td["start"], "end": td["end"],
            "ann_td_pct": td["ann_td_pct"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "monthly_t": td["monthly_t"], "hac_t": td["hac_t"],
        }
    return out


def calendar_year_table(fund: pd.Series, bench: pd.Series) -> pd.DataFrame:
    """Per-calendar-year annualised tracking difference (% / yr). Partial years are
    kept but flagged by their day count — never dropped silently."""
    d = log_diff(fund, bench)
    rows = []
    for y in sorted(set(d.index.year)):
        g = d[d.index.year == y]
        rows.append({"year": int(y), "n_days": int(len(g)),
                     "ann_td_pct": float(g.mean() * TRADING_DAYS * 100.0)})
    return pd.DataFrame(rows).set_index("year")


# --------------------------------------------------------------------------- #
# Synthetic control — the machinery proof (never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame) -> dict:
    """Run both estimators on a synthetic panel with a known planted drag spread.

    On the planted world the *fund-vs-fund* estimator must recover the spread with a
    tight interval, while the *fund-vs-coin* estimator must come back so wide that it
    cannot resolve it — that is the whole methodological claim of the study, proved on
    data whose truth we set. At ``signal_strength=0`` the fund-vs-fund estimate must
    collapse to ~0 (the null).
    """
    ff = tracking_difference(prices["dear"], prices["cheap"])
    fc = tracking_difference(prices["cheap"], prices["coin"])
    ci_ff = bootstrap_td_ci(ff["diff"])
    ci_fc = bootstrap_td_ci(fc["diff"])
    return {
        "fund_vs_fund_pct": ff["ann_td_pct"],
        "fund_vs_fund_ci": (ci_ff["ci_low"], ci_ff["ci_high"]),
        "fund_vs_fund_halfwidth": ci_ff["half_width"],
        "fund_vs_fund_monthly_t": ff["monthly_t"],
        "fund_vs_coin_pct": fc["ann_td_pct"],
        "fund_vs_coin_ci": (ci_fc["ci_low"], ci_fc["ci_high"]),
        "fund_vs_coin_halfwidth": ci_fc["half_width"],
        "fund_vs_coin_monthly_t": fc["monthly_t"],
        "n_days": ff["n_days"],
    }
