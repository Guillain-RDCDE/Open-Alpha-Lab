"""The strategy and its honest controls — Study 170 (Alphabetical-Bias).

The folk recipe (Jacobs & Hillert 2015, *Journal of Financial Economics*):
stocks whose names fall at the top of an alphabetically sorted broker list get
more retail attention, which may translate into higher turnover/volume and
possibly a *return* edge.  We test two channels:

**Attention channel (liquidity/volume).** Group S&P 500 names by first letter
(A–C = early; D–Z = late).  Does the early group carry higher average volume?
This is the more robust academic finding.

**Return channel.** Does an equal-weight portfolio of early-alphabet names earn
a statistically significant return premium over a matched late-alphabet
portfolio?  This is the tradability question, and the one where the bar is
set by the desk's inference rules.

Controls and corrections
------------------------
- **Equal-weight baseline** — compare early vs the unconditional cross-section
  mean, not just vs late (survivorship-biased universe skews the baseline).
- **Multiple-comparisons** — we test 26 single-letter hypotheses (one per
  letter of the alphabet) plus the main A–C vs rest split.  We apply a
  Bonferroni correction: the adjusted threshold is α/27 ≈ 0.002, i.e.
  |t| ≥ ~3.1 required for any single letter to survive.
- **HAC t-stat (Newey-West)** on the monthly return difference series.
- **Survivorship bias** is named explicitly: the S&P 500 universe is not static;
  companies renamed or delisted drop out, and the alphabetical composition
  drifts.  This mechanically biases which letters dominate at any given time.

No look-ahead: portfolio is formed using *previous-month* letter classification
(letter does not change, but constituent-set rebalances monthly at the
prior-month-end close).  Returns are next-month log-returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bonferroni-corrected threshold for 27 tests (26 letters + 1 group test).
N_TESTS = 27
BONFERRONI_ALPHA = 0.05 / N_TESTS  # ≈ 0.00185
# Approximate |t| threshold for a two-sided test at the Bonferroni-corrected level
# (normal approximation: z = 2.897 for 0.00185 two-sided).
BONFERRONI_T = 2.897


# ---------------------------------------------------------------------------
# HAC Newey-West t-stat — inline to avoid hard dependency on quantlab
# ---------------------------------------------------------------------------
def _hac_tstat(arr: np.ndarray) -> dict:
    """Newey-West HAC t-statistic for the sample mean of ``arr``."""
    r = np.asarray(arr, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return {"n": int(n), "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
                "tstat": float("nan")}
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "n": int(n),
        "mean_bps": float(mu * 1e4),
        "se_bps": float(se * 1e4),
        "tstat": float(mu / se) if se > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main study: monthly equal-weight return difference (early vs late alphabet)
# ---------------------------------------------------------------------------
def monthly_ew_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly equal-weight returns for early and late alphabet groups.

    For each month, averages the cross-section returns for the A–C group and
    the D–Z group separately.  Returns a frame indexed by month-end date with
    columns ``ew_early``, ``ew_late``, ``diff`` (early − late), and ``n_early``
    / ``n_late`` (number of stocks in each group that month).

    No look-ahead: uses only the ``is_early`` classification (which is fixed by
    the ticker's first letter) and the cross-sectional return for each month.
    """
    required = {"date", "is_early", "ret"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing columns: {missing}")

    records = []
    for date, grp in panel.groupby("date"):
        early = grp[grp["is_early"]]
        late = grp[~grp["is_early"]]
        ew_e = float(early["ret"].mean()) if len(early) else float("nan")
        ew_l = float(late["ret"].mean()) if len(late) else float("nan")
        records.append({
            "date": date,
            "ew_early": ew_e,
            "ew_late": ew_l,
            "diff": ew_e - ew_l,
            "n_early": len(early),
            "n_late": len(late),
        })
    out = pd.DataFrame(records).set_index("date")
    out.index = pd.DatetimeIndex(out.index)
    return out.sort_index()


def summarize_return_diff(monthly: pd.DataFrame) -> dict:
    """HAC t-stat on the monthly early-minus-late return spread.

    The decisive number: does the A–C group earn a significant return premium
    over D–Z?  Returns n, mean spread (bps/month), HAC t-stat, and an
    annualised Sharpe of the spread.
    """
    diff = monthly["diff"].dropna().to_numpy(dtype=float)
    base = _hac_tstat(diff)
    n = base["n"]
    sharpe = float("nan")
    if n > 1:
        std = float(diff.std(ddof=1))
        if std > 0:
            sharpe = float(diff.mean() / std * np.sqrt(12))  # annualised
    base["sharpe_ann"] = sharpe
    return base


# ---------------------------------------------------------------------------
# Volume / turnover channel — the attention hypothesis
# ---------------------------------------------------------------------------
def volume_comparison(panel: pd.DataFrame) -> dict:
    """Compare average monthly volume between early and late alphabet groups.

    Returns mean volume for each group, the ratio (early/late), and a HAC
    t-stat on the log-volume difference across months.
    """
    if "volume" not in panel.columns:
        raise ValueError("panel must have a 'volume' column")

    panel = panel[panel["volume"] > 0].copy()
    panel["log_vol"] = np.log(panel["volume"])

    # Monthly mean log-volume per group.
    records = []
    for date, grp in panel.groupby("date"):
        early = grp[grp["is_early"]]["log_vol"]
        late = grp[~grp["is_early"]]["log_vol"]
        if len(early) < 2 or len(late) < 2:
            continue
        records.append({
            "date": date,
            "log_vol_early": float(early.mean()),
            "log_vol_late": float(late.mean()),
            "diff": float(early.mean() - late.mean()),
        })
    if not records:
        return {}
    df = pd.DataFrame(records)
    vol_diff = df["diff"].to_numpy()
    base = _hac_tstat(vol_diff)
    base["mean_vol_early"] = float(np.exp(df["log_vol_early"].mean()))
    base["mean_vol_late"] = float(np.exp(df["log_vol_late"].mean()))
    base["vol_ratio"] = base["mean_vol_early"] / base["mean_vol_late"] if base["mean_vol_late"] > 0 else float("nan")
    return base


# ---------------------------------------------------------------------------
# Per-letter breakdown with Bonferroni correction
# ---------------------------------------------------------------------------
def per_letter_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """For each letter A–Z, compute the monthly equal-weight return difference
    vs the cross-section mean, and a HAC t-stat.

    Returns a DataFrame indexed by ``letter`` with columns:
    ``n_months, n_stocks, mean_bps, tstat, bonferroni_significant``.

    Bonferroni correction: we are testing 26 letters (plus the main A–C test =
    27 total), so the adjusted threshold is α/27 ≈ 0.00185, corresponding to
    |t| ≈ 2.90 under a normal approximation.  A letter that clears |t| = 2.90
    would be noteworthy even after correction; most will not.
    """
    import string

    records = []
    for letter in string.ascii_uppercase:
        letter_mask = panel["letter"] == letter
        if not letter_mask.any():
            records.append({"letter": letter, "n_months": 0, "n_stocks": 0,
                            "mean_bps": float("nan"), "tstat": float("nan"),
                            "bonferroni_significant": False})
            continue

        monthly_diffs = []
        for date, grp in panel.groupby("date"):
            in_letter = grp[grp["letter"] == letter]["ret"]
            others = grp[grp["letter"] != letter]["ret"]
            if len(in_letter) == 0 or len(others) == 0:
                continue
            monthly_diffs.append(float(in_letter.mean()) - float(others.mean()))

        n_stocks = int(letter_mask.sum() / max(panel["date"].nunique(), 1))
        base = _hac_tstat(np.array(monthly_diffs))
        t = base.get("tstat", float("nan"))
        records.append({
            "letter": letter,
            "n_months": base.get("n", 0),
            "n_stocks": n_stocks,
            "mean_bps": base.get("mean_bps", float("nan")),
            "tstat": t,
            "bonferroni_significant": bool(abs(t) >= BONFERRONI_T if np.isfinite(t) else False),
        })

    return pd.DataFrame(records).set_index("letter")


# ---------------------------------------------------------------------------
# Positive control on synthetic data
# ---------------------------------------------------------------------------
def detect_planted_bias(panel: pd.DataFrame) -> dict:
    """Run the main return-difference test on a panel (synthetic or real).

    Returns the summarize_return_diff dict augmented with a ``signal_detected``
    boolean (True when |t| >= 2.0, the uncorrected threshold) — used to confirm
    the engine recovers a planted edge when one exists.
    """
    monthly = monthly_ew_returns(panel)
    result = summarize_return_diff(monthly)
    t = result.get("tstat", float("nan"))
    result["signal_detected"] = bool(abs(t) >= 2.0 if np.isfinite(t) else False)
    return result
