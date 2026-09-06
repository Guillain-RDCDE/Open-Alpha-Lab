"""Consistency checks on a market-data feed — Study 971.

Six audits, each with a right answer that does not depend on anybody's opinion:

1. ``resample_agreement`` — compound the daily total-return series into weeks and months and
   compare with the provider's own weekly and monthly bars. These should be the same series.
2. ``reconstruct_total_return`` — rebuild the adjusted series from the as-traded close plus
   the dividend and split events, and compare with the provider's adjusted close. This is the
   single most load-bearing computation in the whole feed: every total-return backtest depends
   on it and almost nobody checks it.
3. ``calendar_gaps`` — count sessions the tape does not have, against the exchange calendar
   inferred from the *union* of all the tickers in the sample. A session that every other US
   equity traded and this one did not is either a halt or a hole.
4. ``split_check`` — for every split event, does the as-traded close jump by the split ratio,
   and does the adjusted close *not*?
5. ``dividend_yield_check`` — do the dividend events sum to something like the fund's known
   yield, and does the total-return-minus-price-return gap match them?
6. ``bar_sanity`` — duplicate dates, non-monotonic index, non-positive prices, absurd daily
   moves.

``audit`` runs all six and returns a tidy frame of findings, each with a severity. The
severities are the study's own judgement, stated once here rather than implied: **error** means
a number computed from this tape will be wrong; **warning** means it may be; **info** means the
disagreement is expected (a weekly bar ends on a different day from the last daily bar of that
week, for instance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SEVERITIES = ("error", "warning", "info")
# Two total-return series that agree to a tenth of a basis point per day are the same series.
TOL_BPS = 1.0


def _cum(close: pd.Series) -> pd.Series:
    return close / close.iloc[0]


def resample_agreement(daily: pd.DataFrame, weekly: pd.DataFrame,
                       monthly: pd.DataFrame) -> pd.DataFrame:
    """Do the provider's weekly and monthly bars match the daily series compounded?

    Compared on **period returns**, aligned on the provider's own period ends, because the two
    series can legitimately have different first observations and a level comparison would then
    report a fault on every row for no reason.
    """
    rows = []
    d = daily["close"].dropna()
    # ``ME`` is the resampling alias (pandas >= 2.2); ``M`` is still the *period* alias.
    for label, frame, rule, freq in (("weekly", weekly, "W", "W"),
                                     ("monthly", monthly, "ME", "M")):
        prov = frame["close"].dropna()
        own = d.resample(rule).last().dropna()
        # Align on the calendar PERIOD, not on the timestamp: a provider stamps a weekly bar
        # with the week's Monday (or, for some tickers, its Friday) and a monthly bar with
        # the first of the month, while a pandas resample stamps the period end. Comparing
        # raw timestamps produces an empty intersection and an audit that reports nothing.
        stamp_weekday = int(pd.Series(prov.index.dayofweek).mode().iloc[0])
        own.index = own.index.to_period(freq)
        prov.index = prov.index.to_period(freq)
        own = own[~own.index.duplicated(keep="last")]
        prov = prov[~prov.index.duplicated(keep="last")]
        # Drop the first and last period of each series: they are partial by construction and
        # would report a fault on every ticker for a reason that is not a fault.
        own, prov = own.iloc[1:-1], prov.iloc[1:-1]

        # The provider's stamp convention is not uniform across tickers, so the honest
        # comparison tries the neighbouring alignments and reports which one fits. A best
        # shift other than zero is itself a finding: it means a naive period join silently
        # compares one week's return with another's.
        best = None
        for shift in (0, -1, 1):
            b_shift = prov.shift(shift).dropna()
            a, b = own.pct_change().dropna().align(b_shift.pct_change().dropna(), join="inner")
            if len(a) < 10:
                continue
            diff = (a - b).abs()
            cand = {"frequency": label, "n_periods": int(len(a)), "shift": shift,
                    "max_abs_diff_bps": float(diff.max() * 1e4),
                    "median_abs_diff_bps": float(diff.median() * 1e4),
                    "n_beyond_10bps": int((diff > 10e-4).sum()),
                    "corr": float(a.corr(b)), "stamp_weekday": stamp_weekday}
            if best is None or cand["median_abs_diff_bps"] < best["median_abs_diff_bps"]:
                best = cand
        if best is not None:
            rows.append(best)
    return pd.DataFrame(rows).set_index("frequency")


WEEKDAY = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def weekly_window_probe(daily: pd.DataFrame, weekly: pd.DataFrame,
                        n_probe: int = 400) -> dict:
    """Which daily session does each weekly bar actually close on?

    The blunt, assumption-free diagnostic, and the one that turns "the weekly series
    disagrees" into a statement about *what the provider did*: for each weekly bar, find the
    daily close that matches it to a part in a million, and record that session's weekday and
    its offset in days from the weekly bar's own label.

    A Monday-stamped bar closing on the following Friday (offset +4) is an ordinary Mon-Fri
    week. A Friday-stamped bar closing on the *next* Thursday (offset +6) is a Fri-Thu week —
    a different window, on the same provider, for a different ticker, with nothing in the
    response to say so.
    """
    d, w = daily["close"].dropna(), weekly["close"].dropna().tail(n_probe)
    offsets, weekdays, hits = {}, {}, 0
    for ts, v in w.items():
        match = d[(d - v).abs() / max(abs(v), 1e-12) < 1e-6]
        if not len(match):
            continue
        hits += 1
        off = int((match.index[-1] - ts).days)
        offsets[off] = offsets.get(off, 0) + 1
        wd = int(match.index[-1].dayofweek)
        weekdays[wd] = weekdays.get(wd, 0) + 1
    if not hits:
        return {"matched": 0, "n_probe": int(len(w)), "modal_offset": None,
                "modal_weekday": None, "window": "unknown"}
    modal_off = max(offsets, key=offsets.get)
    modal_wd = max(weekdays, key=weekdays.get)
    stamp_wd = int(pd.Series(w.index.dayofweek).mode().iloc[0])
    return {"matched": hits, "n_probe": int(len(w)), "modal_offset": modal_off,
            "modal_weekday": WEEKDAY[modal_wd], "stamp_weekday": WEEKDAY[stamp_wd],
            "share_modal": float(offsets[modal_off] / hits),
            "window": f"{WEEKDAY[stamp_wd]} label -> closes {WEEKDAY[modal_wd]} "
                      f"({modal_off:+d} days)"}


def reconstruct_total_return(raw: pd.DataFrame) -> dict:
    """Rebuild the adjusted series from as-traded close + dividends + splits.

    The reconstruction is the textbook one: on a dividend day the holder's wealth grows by
    ``(close + dividend) / close``; on a split day the share count multiplies by the ratio.
    Compounding those factors onto the price return gives the total-return series, which
    should match the provider's ``adj_close`` up to floating point.

    A systematic gap here means one of three things, and the sign says which: dividends
    missing from the events column (reconstruction *below* the provider), a dividend counted
    twice (above), or a split applied on the wrong day (a single enormous outlier).
    """
    if "adj_close" not in raw.columns or "dividends" not in raw.columns:
        return {"available": False}
    df = raw.dropna(subset=["close"]).copy()
    close = df["close"]
    div = df.get("dividends", pd.Series(0.0, index=df.index)).fillna(0.0)
    # NOTE the provider convention: with ``auto_adjust=False`` the close is ALREADY
    # split-adjusted, so the split column must NOT be multiplied in again. Doing so is the
    # classic double-adjustment bug (it manufactures a +100% day at every 2:1 split); the
    # split column is audited separately in :func:`split_check` instead.
    price_ret = close.pct_change().fillna(0.0)
    div_yield = (div / close.shift(1)).fillna(0.0)
    total_ret = price_ret + div_yield
    rebuilt = (1 + total_ret).cumprod()
    provider = _cum(df["adj_close"].dropna())
    a, b = rebuilt.align(provider, join="inner")
    ratio = (a / a.iloc[0]) / (b / b.iloc[0])
    ann = float(ratio.iloc[-1] ** (TRADING_DAYS / len(ratio)) - 1)
    spl = df.get("stock_splits", pd.Series(0.0, index=df.index)).fillna(0.0)
    return {"available": True, "n": int(len(a)),
            "terminal_ratio": float(ratio.iloc[-1]),
            "annualised_gap": ann,
            "max_daily_diff_bps": float(((1 + total_ret) /
                                         (1 + provider.pct_change().fillna(0.0)) - 1
                                         ).abs().max() * 1e4),
            "n_dividends": int((div > 0).sum()),
            "n_splits": int((spl > 0).sum()),
            "total_dividends": float(div.sum())}


def calendar_gaps(daily: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    """Sessions in the reference calendar that this tape does not have (and the reverse)."""
    # Compare only where the two overlap in *time*: a tape whose history starts before the
    # reference calendar does is not full of "extra" sessions, it is simply older.
    lo = max(daily.index.min(), calendar.min())
    hi = min(daily.index.max(), calendar.max())
    have = daily.index[(daily.index >= lo) & (daily.index <= hi)]
    cal = calendar[(calendar >= lo) & (calendar <= hi)]
    missing = cal.difference(have)
    extra = have.difference(cal)
    return {"n_sessions": int(len(have)), "n_calendar": int(len(cal)),
            "n_missing": int(len(missing)), "n_extra": int(len(extra)),
            "missing_dates": [str(d.date()) for d in missing[:10]],
            "extra_dates": [str(d.date()) for d in extra[:10]],
            "coverage": float(1 - len(missing) / max(len(cal), 1))}


def build_calendar(daily_by_ticker: dict) -> pd.DatetimeIndex:
    """The reference trading calendar: any date at least half the tapes traded on."""
    counts: dict = {}
    for df in daily_by_ticker.values():
        for d in df.index:
            counts[d] = counts.get(d, 0) + 1
    need = max(1, len(daily_by_ticker) // 2)
    return pd.DatetimeIndex(sorted(d for d, c in counts.items() if c >= need))


def split_check(raw: pd.DataFrame, adjusted: pd.DataFrame,
                tolerance: float = 0.25) -> pd.DataFrame:
    """On a split date, **neither** series may jump by the split ratio.

    This is the check whose expected answer surprises people. With ``auto_adjust=False`` the
    provider's OHLC is already **split-adjusted** — only dividends separate it from
    ``adj_close`` (a convention pinned empirically in ``quantlab/data.py``). So a correct feed
    shows a 4:1 split as an ordinary day in *both* series, and the failure mode to hunt is the
    opposite of the intuitive one: a price that *does* divide by the ratio is a tape where the
    adjustment was not applied, and a backtest reading it sees a −75% day.

    ``ratio_move`` is the move a *non-adjusted* tape would print; ``ok`` is True when the day
    looks ordinary in both series.
    """
    if "stock_splits" not in raw.columns:
        return pd.DataFrame()
    events = raw.index[raw["stock_splits"].fillna(0) > 0]
    rows = []
    for d in events:
        i = raw.index.get_loc(d)
        if i == 0:
            continue
        ratio = float(raw["stock_splits"].iloc[i])
        raw_move = float(raw["close"].iloc[i] / raw["close"].iloc[i - 1])
        j = adjusted.index.get_indexer([d])[0]
        adj_move = (float(adjusted["close"].iloc[j] / adjusted["close"].iloc[j - 1])
                    if j > 0 else np.nan)
        looks_unadjusted = abs(raw_move - 1.0 / ratio) < 0.05
        rows.append({"date": str(d.date()), "ratio": ratio, "raw_move": raw_move,
                     "adjusted_move": adj_move,
                     "raw_ok": bool(abs(raw_move - 1.0) < tolerance),
                     "adjusted_ok": bool(np.isfinite(adj_move) and abs(adj_move - 1.0) < tolerance),
                     "looks_unadjusted": bool(looks_unadjusted)})
    return pd.DataFrame(rows)


def dividend_yield_check(raw: pd.DataFrame, adjusted: pd.DataFrame) -> dict:
    """Does the total-return-minus-price-return gap equal the dividends the feed reports?"""
    if "dividends" not in raw.columns:
        return {"available": False}
    px = raw["close"].dropna()
    tr = adjusted["close"].dropna()
    a, b = px.align(tr, join="inner")
    years = len(a) / TRADING_DAYS
    price_cagr = float((a.iloc[-1] / a.iloc[0]) ** (1 / years) - 1)
    total_cagr = float((b.iloc[-1] / b.iloc[0]) ** (1 / years) - 1)
    reported = float((raw["dividends"].fillna(0.0) / raw["close"]).sum() / years)
    return {"available": True, "price_cagr": price_cagr, "total_cagr": total_cagr,
            "implied_yield": total_cagr - price_cagr, "reported_yield": reported,
            "gap": (total_cagr - price_cagr) - reported, "years": float(years)}


def bar_sanity(df: pd.DataFrame, threshold: float = 0.40) -> dict:
    """Duplicates, ordering, non-positive prices and moves too large to be a market."""
    close = df["close"]
    r = close.pct_change().dropna()
    return {"n": int(len(df)), "duplicate_dates": int(df.index.duplicated().sum()),
            "monotonic": bool(df.index.is_monotonic_increasing),
            "non_positive": int((close <= 0).sum()),
            "nan_close": int(close.isna().sum()),
            "moves_beyond_threshold": int((r.abs() > threshold).sum()),
            "worst_move": float(r.abs().max()) if len(r) else np.nan}


# --------------------------------------------------------------------------- #
# The audit
# --------------------------------------------------------------------------- #
def audit(frames: dict, calendar: pd.DatetimeIndex | None = None,
          ticker: str = "?") -> pd.DataFrame:
    """Run every check on one ticker's four frames and return tidy findings."""
    findings = []

    def add(check, severity, detail, value=np.nan):
        findings.append({"ticker": ticker, "check": check, "severity": severity,
                         "detail": detail, "value": value})

    ra = resample_agreement(frames["daily_tr"], frames["weekly"], frames["monthly"])
    weekday = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday",
               5: "Saturday", 6: "Sunday"}
    for freq, row in ra.iterrows():
        sev = ("error" if row["max_abs_diff_bps"] > 500 else
               "warning" if row["n_beyond_10bps"] > 0 else "info")
        add(f"resample:{freq}", sev,
            f"{int(row['n_beyond_10bps'])} of {int(row['n_periods'])} periods differ by more "
            f"than 10 bps; worst {row['max_abs_diff_bps']:.1f} bps",
            row["max_abs_diff_bps"])
        if freq == "weekly":
            probe = weekly_window_probe(frames["daily_tr"], frames["weekly"])
            sev = "warning" if probe.get("modal_offset") not in (3, 4) else "info"
            add("weekly_window", sev,
                f"{probe['window']} on {probe['matched']}/{probe['n_probe']} probed bars "
                f"({probe.get('share_modal', 0):.0%} agree)",
                float(probe.get("modal_offset") or np.nan))
        if int(row["shift"]) != 0:
            add(f"stamp_convention:{freq}", "warning",
                f"bars are stamped on {weekday.get(int(row['stamp_weekday']), '?')} and match "
                f"the daily tape only after shifting by {int(row['shift'])} period — a naive "
                f"join on the label compares the wrong period",
                float(row["shift"]))

    rec = reconstruct_total_return(frames["daily_raw"])
    if rec.get("available"):
        sev = ("error" if abs(rec["annualised_gap"]) > 0.005 else
               "warning" if abs(rec["annualised_gap"]) > 0.001 else "info")
        add("reconstruct_total_return", sev,
            f"rebuilt series differs from the provider's adjusted close by "
            f"{rec['annualised_gap']:+.3%}/yr over {rec['n']} sessions "
            f"({rec['n_dividends']} dividends, {rec['n_splits']} splits)",
            rec["annualised_gap"])

    if calendar is not None:
        cg = calendar_gaps(frames["daily_tr"], calendar)
        sev = ("error" if cg["n_missing"] > 10 else
               "warning" if cg["n_missing"] > 0 else "info")
        add("calendar_gaps", sev,
            f"{cg['n_missing']} sessions missing of {cg['n_calendar']} in the reference "
            f"calendar (coverage {cg['coverage']:.4%}); {cg['n_extra']} dates not in it",
            cg["n_missing"])

    sc = split_check(frames["daily_raw"], frames["daily_tr"])
    if len(sc):
        bad = sc[~sc["raw_ok"] | ~sc["adjusted_ok"]]
        add("split_check", "error" if len(bad) else "info",
            f"{len(sc)} split events, {len(bad)} inconsistent", float(len(bad)))

    dy = dividend_yield_check(frames["daily_raw"], frames["daily_tr"])
    if dy.get("available"):
        sev = ("error" if abs(dy["gap"]) > 0.01 else
               "warning" if abs(dy["gap"]) > 0.002 else "info")
        add("dividend_yield", sev,
            f"total-minus-price CAGR is {dy['implied_yield']:.2%}/yr against "
            f"{dy['reported_yield']:.2%}/yr of reported dividends (gap {dy['gap']:+.2%})",
            dy["gap"])

    bs = bar_sanity(frames["daily_tr"])
    problems = bs["duplicate_dates"] + bs["non_positive"] + bs["nan_close"]
    add("bar_sanity", "error" if problems or not bs["monotonic"] else "info",
        f"{bs['duplicate_dates']} duplicate dates, {bs['non_positive']} non-positive closes, "
        f"{bs['moves_beyond_threshold']} moves beyond 40%, worst {bs['worst_move']:.1%}",
        float(problems))
    return pd.DataFrame(findings)


def severity_counts(findings: pd.DataFrame) -> pd.Series:
    """How many findings of each severity — the headline of the audit."""
    return findings["severity"].value_counts().reindex(list(SEVERITIES)).fillna(0).astype(int)


def backtest_impact(daily: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    """Does the disagreement change a number anybody would publish?

    A deliberately ordinary calculation — the annualised return, volatility and Sharpe of a
    buy-and-hold position — computed from the daily tape and from the provider's own weekly
    bars. If a feed is self-consistent these agree to the third decimal; if they do not, the
    difference is the size of the error a reader would inherit without ever knowing.
    """
    d = daily["close"].dropna()
    w = weekly["close"].dropna()
    rd, rw = d.pct_change().dropna(), w.pct_change().dropna()
    # Calendar years from the index, not observation counts: the two series end on different
    # days, and dividing by 252 and by 52 respectively would introduce a spurious CAGR gap
    # before any data fault had a chance to.
    yr_d = (d.index[-1] - d.index[0]).days / 365.25
    yr_w = (w.index[-1] - w.index[0]).days / 365.25
    out = {
        "cagr_daily": float((d.iloc[-1] / d.iloc[0]) ** (1 / yr_d) - 1),
        "cagr_weekly": float((w.iloc[-1] / w.iloc[0]) ** (1 / yr_w) - 1),
        "vol_daily": float(rd.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "vol_weekly": float(rw.std(ddof=1) * np.sqrt(52)),
    }
    out["sharpe_daily"] = out["cagr_daily"] / out["vol_daily"] if out["vol_daily"] else np.nan
    out["sharpe_weekly"] = out["cagr_weekly"] / out["vol_weekly"] if out["vol_weekly"] else np.nan
    out["cagr_gap"] = out["cagr_daily"] - out["cagr_weekly"]
    out["vol_gap"] = out["vol_daily"] - out["vol_weekly"]
    out["sharpe_gap"] = out["sharpe_daily"] - out["sharpe_weekly"]
    return out


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal** (are there real inconsistencies?): **Real** if any check returns an *error*
      severity on any ticker; **Weak** if only warnings; **None** if everything is clean.
    - **Usefulness** (would it change a published result?): **Useful** if the largest
      daily-versus-weekly Sharpe discrepancy exceeds 0.05 or the reconstruction gap exceeds
      0.5%/yr — both are large enough to move a conclusion; **Fragile** at a tenth of that;
      **Mirage** below.
    """
    signal = ("Real" if h["n_errors"] > 0 else
              ("Weak" if h["n_warnings"] > 0 else "None"))
    big = h["max_sharpe_gap"] >= 0.05 or h["max_reconstruction_gap"] >= 0.005
    small = h["max_sharpe_gap"] >= 0.005 or h["max_reconstruction_gap"] >= 0.0005
    trad = "Useful" if big else ("Fragile" if small else "Mirage")
    return {
        "signal": signal,
        "signal_why": (
            f"The audit ran **{h['n_checks']}** checks across {h['n_tickers']} tickers and "
            f"returned **{h['n_errors']} errors** and **{h['n_warnings']} warnings**. The "
            f"weekly and monthly bars compound from the daily ones to within "
            f"**{h['max_resample_bps']:.0f} bps** at worst; rebuilding the total-return series "
            f"from price plus dividends and splits reproduces the provider's own adjusted "
            f"close to **{h['max_reconstruction_gap']:+.3%}/yr** at worst "
            f"({h['worst_reconstruction_ticker']}); and the reference calendar shows "
            f"**{h['total_missing_sessions']}** missing sessions in total."),
        "trad": trad,
        "trad_why": (
            f"The same buy-and-hold statistic computed from the daily tape and from the "
            f"provider's own weekly bars differs by up to **{h['max_sharpe_gap']:.3f}** of "
            f"Sharpe and **{h['max_cagr_gap']:+.2%}/yr** of CAGR "
            f"({h['worst_backtest_ticker']}) — most of which is the arithmetic of measuring "
            f"volatility at a different frequency rather than a data fault. The fault that "
            f"*would* change a published number is the reconstruction gap, and the audit "
            f"prices it at {h['max_reconstruction_gap']:+.3%}/yr."),
        "one_sentence": (
            f"A free daily feed is much better than its reputation — the weekly and monthly "
            f"bars agree with the daily ones to a few basis points and the adjusted close is "
            f"reconstructible to {h['max_reconstruction_gap']:+.3%}/yr — but it is not perfect, "
            f"and the checks that catch the exceptions cost six functions and a test suite."),
    }
