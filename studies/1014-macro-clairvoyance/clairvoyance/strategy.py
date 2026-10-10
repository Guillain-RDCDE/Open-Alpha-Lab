"""What perfect macro foresight is worth — Study 1014.

We hand a hypothetical investor an oracle. Each quarter, before the quarter starts, the oracle
whispers the *final-vintage* value of one macro variable — GDP growth, the change in
unemployment, CPI inflation or the change in the T-bill rate — for some quarter relative to the
one the investor is about to hold. The investor is in equities for the coming quarter if the
whispered number is on the "good" side of its own past-only median, and in T-bills otherwise.

The single most important parameter is the **information horizon** ``h``: the oracle reveals
``x_{t+h}`` and the investor holds the market over quarter ``t``.

=====  ======================================================================================
h=+2   the oracle sees the economy **two quarters beyond** the holding quarter
h=+1   **one quarter beyond** the holding quarter
h=0    **the holding quarter itself** — a perfect forecast of the coming quarter's macro
       print. This is the macro-forecasting industry's product, made flawless; it is the
       study's headline oracle.
h=−1   the quarter that just ended — what a real investor has after the release lag
       (generous: the advance GDP estimate actually arrives about four weeks into quarter t)
h=−2   strict release lag: only data comfortably published before the quarter starts
=====  ======================================================================================

**Execution lag.** There is exactly one: the position for quarter ``t`` is fixed at the close of
quarter ``t−1`` and earns quarter ``t``'s return. All the look-ahead lives in the oracle's
``h``; nothing else is shifted. Costs are charged one-way on the traded fraction of NAV
(``|pos_t − pos_{t−1}|``) in the quarter the trade happens.

**Inference.** Every comparison is excess-of-cash against excess-of-cash (out of the market the
book earns the bill rate, i.e. zero excess). The Sharpe difference against buy-and-hold is
tested two ways that do not lean on normality:

- a **circular block bootstrap** of the paired quarterly returns (blocks of four quarters),
  centred so the p-value tests ΔSharpe ≤ 0;
- a **rotation placebo**: the oracle's own position series is circularly shifted against the
  returns by every offset of at least four quarters. A rotated oracle keeps its duty cycle and
  persistence but loses its alignment with the market, so it is the right null for "is this
  macro signal special, or would any signal that is in the market ~half the time do as well?".

A Newey-West *t* on the active return (strategy minus buy-and-hold) is reported alongside.

**The ceiling.** An investor who knows the *market's* own sign each quarter — the perfect
market oracle — sets the scale. The share of the ceiling's Sharpe gain that a macro oracle
captures is the study's measure of how much of "knowing the future" macro knowledge actually is.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from quantlab.analytics import mean_tstat_hac  # noqa: E402

from .data import GOOD_SIGN, QUARTERS_PER_YEAR  # noqa: E402

HORIZONS = (-2, -1, 0, 1, 2)
WARMUP = 20            # quarters of history before the past-only median is trusted
BLOCK = 4              # bootstrap block, quarters
MIN_SHIFT = 4          # smallest rotation offset in the placebo
COST_BPS = 10.0        # one-way cost per unit of NAV traded, headline
COMBINED = ("g", "du", "infl")


# --------------------------------------------------------------------------- #
# Small statistics
# --------------------------------------------------------------------------- #
def ann_sharpe(r) -> float:
    """Annualised Sharpe of quarterly excess returns: mean / sd · √4."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(QUARTERS_PER_YEAR)) if sd > 0 else float("nan")


def ols_hac(y, x, lags: int = 4) -> dict:
    """Slope of ``y`` on ``x`` (with intercept) and its Newey-West *t*; also the correlation.

    Used for the lead-lag table, where ``y`` is the quarterly excess return and ``x`` a macro
    variable shifted by ``k`` quarters. Bartlett kernel, ``lags`` quarters.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = y.size
    if n < 20 or np.std(x) == 0:
        return {"slope": np.nan, "t": np.nan, "corr": np.nan, "n": int(n)}
    A = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    u = y - A @ beta
    XtX_inv = np.linalg.inv(A.T @ A)
    Au = A * u[:, None]
    S = Au.T @ Au
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Au[l:].T @ Au[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return {"slope": float(beta[1]), "t": float(beta[1] / se) if se > 0 else np.nan,
            "corr": float(np.corrcoef(x, y)[0, 1]), "n": int(n)}


# --------------------------------------------------------------------------- #
# (a) Who leads whom
# --------------------------------------------------------------------------- #
def lead_lag(panel: pd.DataFrame, var: str, ks=range(-4, 5), lags: int = 4) -> pd.DataFrame:
    """Correlation of the quarter-``t`` excess return with ``var`` at quarter ``t+k``.

    ``k > 0``: the macro print comes *after* the return (stocks lead the economy).
    ``k < 0``: the macro print came *before* (the economy leads stocks — the tradable case).
    Each row carries the correlation and the Newey-West *t* of the slope.
    """
    rows = []
    for k in ks:
        d = ols_hac(panel["ex"].to_numpy(), panel[var].shift(-k).to_numpy(), lags=lags)
        rows.append({"k": int(k), **d})
    return pd.DataFrame(rows).set_index("k")


# --------------------------------------------------------------------------- #
# (b) The oracles
# --------------------------------------------------------------------------- #
def oracle_position(panel: pd.DataFrame, var: str, h: int, warmup: int = WARMUP) -> pd.Series:
    """1 = in equities for quarter ``t``, 0 = in bills, NaN = undefined (warm-up / no data).

    The oracle reveals ``x_{t+h}`` (signed so that larger is "good for stocks"). The threshold
    is the **expanding, past-only median** of the same signed variable, using only quarters
    that had been realised before the revealed one *and* before the decision: for ``h ≥ 0``
    the median of ``x_s, s ≤ t−1``; for ``h < 0`` the median of ``x_s, s < t+h``.
    """
    x = GOOD_SIGN[var] * panel[var].astype(float)
    revealed = x.shift(-h)
    med = x.expanding(min_periods=warmup).median()
    thresh = med.shift(1 - min(h, 0))
    pos = (revealed > thresh).astype(float)
    pos[revealed.isna() | thresh.isna()] = np.nan
    return pos.rename(f"{var}@{h:+d}")


def combined_position(panel: pd.DataFrame, h: int, vars_=COMBINED,
                      warmup: int = WARMUP) -> pd.Series:
    """Majority vote of the single-variable oracles (in equities if ≥ half say "good")."""
    P = pd.concat([oracle_position(panel, v, h, warmup) for v in vars_], axis=1)
    pos = (P.mean(axis=1) > 0.5).astype(float)
    pos[P.isna().any(axis=1)] = np.nan
    return pos.rename(f"combined@{h:+d}")


def market_oracle(panel: pd.DataFrame) -> pd.Series:
    """The ceiling: in equities exactly in the quarters whose excess return is positive."""
    return (panel["ex"] > 0).astype(float).rename("market oracle")


def book(pos: pd.Series, ex: pd.Series, cost_bps: float = COST_BPS) -> pd.Series:
    """Quarterly **net** excess return of a long/bills book.

    ``pos_t · ex_t − cost · |pos_t − pos_{t−1}|`` with cost one-way in bps of NAV traded. The
    first quarter of the window is charged for entering from cash if it starts invested.
    """
    pos = pos.astype(float)
    turn = pos.diff().abs()
    turn.iloc[0] = abs(pos.iloc[0])
    return (pos * ex - cost_bps / 1e4 * turn).rename(pos.name)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def sharpe_diff_bootstrap(a, b, n_boot: int = 2000, block: int = BLOCK,
                          seed: int = 1014) -> dict:
    """Circular block bootstrap of ``Sharpe(a) − Sharpe(b)`` on paired quarters.

    Returns the point difference, a 95% percentile interval and a one-sided p-value for
    ``H0: ΔSharpe ≤ 0`` from the *centred* bootstrap distribution
    (share of ``d* − d̂ ≥ d̂``).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size
    d_hat = ann_sharpe(a) - ann_sharpe(b)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        draws[i] = ann_sharpe(a[idx]) - ann_sharpe(b[idx])
    draws = draws[np.isfinite(draws)]
    p = float(np.mean(draws - d_hat >= d_hat)) if draws.size else np.nan
    return {"dsharpe": float(d_hat), "ci_lo": float(np.quantile(draws, 0.025)),
            "ci_hi": float(np.quantile(draws, 0.975)), "p": p, "n_boot": int(draws.size)}


def rotation_test(pos: pd.Series, ex: pd.Series, cost_bps: float = COST_BPS,
                  min_shift: int = MIN_SHIFT) -> dict:
    """Rotation placebo: the oracle's positions circularly shifted against the returns.

    Every offset ``s`` with ``min_shift ≤ s ≤ n − min_shift`` is used (exhaustive, so no seed).
    p = (1 + #{rotated Sharpe ≥ actual}) / (1 + #rotations).
    """
    p_arr = pos.to_numpy(dtype=float)
    e = ex.to_numpy(dtype=float)
    n = p_arr.size
    actual = ann_sharpe(book(pos, ex, cost_bps))
    null = []
    for s in range(min_shift, n - min_shift + 1):
        rp = pd.Series(np.roll(p_arr, s), index=pos.index)
        null.append(ann_sharpe(book(rp, pd.Series(e, index=pos.index), cost_bps)))
    null = np.asarray(null)
    return {"sharpe": actual, "null_mean": float(np.nanmean(null)),
            "null_q95": float(np.nanquantile(null, 0.95)),
            "p": float((1 + np.sum(null >= actual)) / (1 + null.size)),
            "n_rot": int(null.size), "null": null}


def evaluate(pos: pd.Series, panel: pd.DataFrame, cost_bps: float = COST_BPS,
             n_boot: int = 2000, seed: int = 1014, ceiling_sharpe: float | None = None,
             with_rotation: bool = True) -> dict:
    """Everything the desk wants to know about one oracle on one window.

    ``pos`` must already be restricted to the evaluation window (no NaN).
    """
    ex = panel["ex"].reindex(pos.index)
    strat = book(pos, ex, cost_bps)
    bh = ex
    out = {"name": pos.name, "n": int(len(pos)),
           "sharpe": ann_sharpe(strat), "bh_sharpe": ann_sharpe(bh),
           "mean_ann": float(strat.mean() * QUARTERS_PER_YEAR),
           "bh_mean_ann": float(bh.mean() * QUARTERS_PER_YEAR),
           "vol_ann": float(strat.std(ddof=1) * np.sqrt(QUARTERS_PER_YEAR)),
           "exposure": float(pos.mean()),
           "switches_per_year": float(pos.diff().abs().sum() / len(pos) * QUARTERS_PER_YEAR),
           "hit_rate": float(((pos > 0.5) == (ex > 0)).mean())}
    bt = sharpe_diff_bootstrap(strat.to_numpy(), bh.to_numpy(), n_boot=n_boot, seed=seed)
    out.update({"dsharpe": bt["dsharpe"], "ci_lo": bt["ci_lo"], "ci_hi": bt["ci_hi"],
                "boot_p": bt["p"]})
    hac = mean_tstat_hac((strat - bh).to_numpy(), lags=4)
    out["active_t"] = float(hac["tstat"])
    out["active_ann"] = float((strat - bh).mean() * QUARTERS_PER_YEAR)
    if with_rotation:
        rt = rotation_test(pos, ex, cost_bps)
        out["rot_p"] = rt["p"]
        out["rot_q95"] = rt["null_q95"]
    if ceiling_sharpe is not None:
        span = ceiling_sharpe - out["bh_sharpe"]
        out["capture"] = float((out["sharpe"] - out["bh_sharpe"]) / span) if span > 0 else np.nan
    return out


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def all_positions(panel: pd.DataFrame, horizons=HORIZONS, warmup: int = WARMUP) -> pd.DataFrame:
    """Every oracle (four variables + combined) at every horizon, as one frame."""
    cols = {}
    for h in horizons:
        for v in GOOD_SIGN:
            p = oracle_position(panel, v, h, warmup)
            cols[p.name] = p
        c = combined_position(panel, h, warmup=warmup)
        cols[c.name] = c
    return pd.DataFrame(cols)


def common_window(P: pd.DataFrame) -> pd.DatetimeIndex:
    """The quarters on which *every* oracle is defined — one window for every comparison."""
    return P.dropna(how="any").index


def horizon_table(panel: pd.DataFrame, horizons=HORIZONS, cost_bps: float = COST_BPS,
                  n_boot: int = 2000, seed: int = 1014, warmup: int = WARMUP,
                  with_rotation: bool = True) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Sharpe, ΔSharpe vs buy-and-hold and both p-values for every oracle × horizon."""
    P = all_positions(panel, horizons, warmup)
    win = common_window(P)
    sub = panel.loc[win]
    ceil = ann_sharpe(book(market_oracle(sub), sub["ex"], cost_bps))
    rows = []
    for name in P.columns:
        var, h = name.split("@")
        e = evaluate(P.loc[win, name], sub, cost_bps, n_boot, seed, ceil, with_rotation)
        rows.append({"var": var, "h": int(h), **e})
    return pd.DataFrame(rows), win


def cost_sweep(pos: pd.Series, panel: pd.DataFrame, costs=(0, 10, 25, 50),
               n_boot: int = 1000, seed: int = 1014) -> pd.DataFrame:
    """The same oracle charged 0 → 50 bps one-way."""
    rows = []
    for c in costs:
        e = evaluate(pos, panel, float(c), n_boot, seed, with_rotation=False)
        rows.append({"cost_bps": c, "sharpe": e["sharpe"], "dsharpe": e["dsharpe"],
                     "boot_p": e["boot_p"]})
    return pd.DataFrame(rows).set_index("cost_bps")


def subperiods(pos: pd.Series, panel: pd.DataFrame, split: str = "1986-12-31",
               cost_bps: float = COST_BPS, n_boot: int = 1000,
               seed: int = 1014) -> pd.DataFrame:
    """The oracle in two halves of the window, split at a round date fixed in advance."""
    rows = []
    for lab, m in (("first half", pos.index <= pd.Timestamp(split)),
                   ("second half", pos.index > pd.Timestamp(split))):
        p = pos[m]
        e = evaluate(p, panel, cost_bps, n_boot, seed, with_rotation=False)
        rows.append({"period": f"{lab} ({p.index[0].year}-{p.index[-1].year})",
                     "n": e["n"], "sharpe": e["sharpe"], "bh_sharpe": e["bh_sharpe"],
                     "dsharpe": e["dsharpe"], "boot_p": e["boot_p"]})
    return pd.DataFrame(rows).set_index("period")


def ceiling_ladder(panel: pd.DataFrame, win: pd.DatetimeIndex, cost_bps: float = COST_BPS,
                   noise=(0.0, 0.1, 0.2, 0.3, 0.4), seed: int = 1014) -> pd.DataFrame:
    """The perfect market oracle, then the same oracle wrong a fraction of the time.

    Locates the macro oracles on a scale from "knows the market" to "coin flip" by hit rate.
    Each noisy row averages 200 random corruption patterns (deterministic seed).
    """
    sub = panel.loc[win]
    truth = market_oracle(sub)
    bh = ann_sharpe(sub["ex"])
    rng = np.random.default_rng(seed)
    rows = []
    for q in noise:
        shs = []
        reps = 1 if q == 0 else 200
        for _ in range(reps):
            flip = rng.random(len(truth)) < q
            p = truth.copy()
            p[flip] = 1.0 - p[flip]
            shs.append(ann_sharpe(book(p, sub["ex"], cost_bps)))
        rows.append({"error_rate": q, "hit_rate": 1 - q, "sharpe": float(np.mean(shs)),
                     "dsharpe": float(np.mean(shs) - bh)})
    return pd.DataFrame(rows).set_index("error_rate")


def synthetic_power(strengths=(1.0, 0.0), seeds=range(1014, 1024), n_quarters: int = 400,
                    n_boot: int = 300, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Positive control and null, across seeds: how often does the GDP h=+1 oracle clear
    both nulls, and how often is the lead-lag *t* at k=+1 above 2?

    A single seed is an anecdote; the rejection *rate* is the property that matters — high
    when the lead is planted, near the nominal level when it is not.
    """
    from .data import synthetic_panel
    rows = []
    for s in strengths:
        for seed in seeds:
            p, _ = synthetic_panel(n_quarters=n_quarters, signal_strength=s, seed=seed)
            pos = oracle_position(p, "g", 1).dropna()
            e = evaluate(pos, p, cost_bps, n_boot=n_boot, seed=seed)
            ll = ols_hac(p["ex"].to_numpy(), p["g"].shift(-1).to_numpy())
            rows.append({"signal_strength": float(s), "seed": int(seed),
                         "sharpe": e["sharpe"], "bh_sharpe": e["bh_sharpe"],
                         "dsharpe": e["dsharpe"], "boot_p": e["boot_p"], "rot_p": e["rot_p"],
                         "passes": passes(e), "lead_t": ll["t"]})
    return pd.DataFrame(rows)


def lead_lag_size(n_seeds: int = 300, n_quarters: int = 400, seed0: int = 2000) -> dict:
    """Empirical size of the lead-lag Newey-West *t* (k=+1) under the synthetic null.

    The GDP series is autocorrelated, so a naive *t* could over-reject; this checks that the
    HAC version rejects at |t| > 2 close to the nominal 5% when returns are independent.
    """
    from .data import synthetic_panel
    ts = []
    for k in range(n_seeds):
        p, _ = synthetic_panel(n_quarters=n_quarters, signal_strength=0.0, seed=seed0 + k)
        ts.append(ols_hac(p["ex"].to_numpy(), p["g"].shift(-1).to_numpy())["t"])
    ts = np.asarray(ts)
    return {"n_seeds": int(n_seeds), "reject_rate": float(np.mean(np.abs(ts) > 2.0)),
            "sd_t": float(np.std(ts))}


# --------------------------------------------------------------------------- #
# Verdict — thresholds fixed before the real run
# --------------------------------------------------------------------------- #
def passes(e: dict, alpha: float = 0.05) -> bool:
    """An oracle 'works' iff it beats buy-and-hold net of costs on BOTH nulls at ``alpha``."""
    return bool(e["dsharpe"] > 0 and e["boot_p"] < alpha and e.get("rot_p", 1.0) < alpha)


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    Inputs (``h``): ``head`` = the combined oracle at ``h=0`` (perfect forecast of the coming
    quarter's GDP, unemployment and inflation), ``lead`` = the combined oracle at ``h=+1``,
    ``real`` = the combined oracle at ``h=−1`` (release lag), each an :func:`evaluate` dict net
    of the headline cost; ``ll_g`` = the GDP lead-lag table as ``{k: {corr, t}}``; the
    ceiling Sharpe ``ceiling``.

    - **Signal** — is perfect macro foresight worth something for timing equities?
      **Real** if the headline oracle passes both nulls (bootstrap and rotation, p < 0.05);
      **Mixed** if the headline fails but the one-quarter-further oracle passes — the verdict
      then genuinely splits by horizon; **Weak** if the headline beats buy-and-hold with
      either p < 0.10; **None** otherwise.
    - **Tradability** — what survives once you only know what has been released? Capped at
      **Fragile**: even the release-lag leg uses final-vintage data nobody had at the time, so
      this tape cannot certify an investable edge. **Fragile** if the release-lag oracle beats
      buy-and-hold net of costs with bootstrap p < 0.10; **Mirage** otherwise.
    - **Third axis, "Stocks lead the economy?"** — **Confirmed** if the excess return's
      correlation with GDP growth one or two quarters *later* has a Newey-West *t* ≥ 2 and
      exceeds the same-quarter correlation; **Not supported** otherwise.
    """
    head, lead, real = h["head"], h["lead"], h["real"]
    if passes(head):
        signal = "Real"
    elif passes(lead):
        signal = "Mixed"
    elif head["dsharpe"] > 0 and min(head["boot_p"], head.get("rot_p", 1.0)) < 0.10:
        signal = "Weak"
    else:
        signal = "None"
    trad = "Fragile" if (real["dsharpe"] > 0 and real["boot_p"] < 0.10) else "Mirage"
    ll = h["ll_g"]
    best_k = max((1, 2), key=lambda k: ll[k]["corr"])
    third = ("Confirmed" if (ll[best_k]["t"] >= 2.0 and ll[best_k]["corr"] > ll[0]["corr"])
             else "Not supported")

    def _fmt(e):
        return (f"Sharpe **{e['sharpe']:.2f}** vs {e['bh_sharpe']:.2f} buy-and-hold, "
                f"ΔSharpe {e['dsharpe']:+.2f}, bootstrap p = {e['boot_p']:.2f}, "
                f"rotation p = {e['rot_p']:.2f}")

    sig_why = (
        f"A flawless forecast of the **coming quarter's** GDP growth, unemployment change and "
        f"inflation (combined oracle, h = 0) timed the market to {_fmt(head)} — "
        + ("enough to clear both nulls. " if signal == "Real" else
           "indistinguishable from simply holding the market. ")
        + f"The value lives further out: knowing the quarter *after* the holding quarter "
        f"(h = +1) gives {_fmt(lead)}"
        + (f", and GDP alone at h = +1 gives {_fmt(h['g_lead'])}" if "g_lead" in h else "")
        + ". The market spends each quarter pricing the next one, so a perfect forecast of the "
        f"quarter you are in arrives already priced. Stamped on the pre-registered headline "
        f"(h = 0); the one-quarter-beyond oracle "
        + ("also failed to clear both nulls at 5%, so the split by horizon is suggestive, "
           "not certified." if not passes(lead) else
           "cleared both nulls, so the verdict splits by horizon."))
    strict = h.get("strict")
    trad_why = (
        f"What a real investor actually has is last quarter's print after the release lag "
        f"(h = −1) — and even that on revised data nobody saw at the time. Net of "
        f"{h['cost_bps']:.0f} bps a switch it gave {_fmt(real)}"
        + (f"; one more quarter of lag (h = −2, data comfortably published before the "
           f"quarter starts) gives {_fmt(strict)}" if strict is not None else "")
        + f". The perfect *market* oracle reaches **{h['ceiling']:.2f}**; the headline macro "
        f"oracle captures **{head['capture']:.0%}** of the gap between buy-and-hold and that "
        f"ceiling, the one-quarter-beyond oracle {lead['capture']:.0%}. "
        + ("A thin, lag-sensitive edge on final-vintage data is the definition of Fragile, "
           "and the vintage problem caps this stamp there whatever the numbers say."
           if trad == "Fragile" else
           "Nothing that is actually knowable survives, and final-vintage data would cap the "
           "stamp at Fragile even if it had."))
    third_why = (
        f"The excess return's correlation with real GDP growth is "
        f"{ll[0]['corr']:+.2f} in the same quarter (t = {ll[0]['t']:+.1f}) but "
        f"{ll[1]['corr']:+.2f} one quarter later (t = {ll[1]['t']:+.1f}) and "
        f"{ll[2]['corr']:+.2f} two quarters later (t = {ll[2]['t']:+.1f}).")
    if signal in ("Real", "Weak"):
        one = (f"A flawless forecast of the coming quarter's GDP, jobs and inflation timed US "
               f"stocks to a Sharpe of {head['sharpe']:.2f} against {head['bh_sharpe']:.2f} "
               f"for buying and holding — worth something, but a fraction of the "
               f"{h['ceiling']:.2f} that knowing the market itself would buy.")
    else:
        one = (f"Even a flawless forecast of the coming quarter's GDP, jobs and inflation "
               f"would have timed US stocks to a Sharpe of {head['sharpe']:.2f} against "
               f"{head['bh_sharpe']:.2f} for buying and holding, because the market had "
               f"already moved on to pricing the quarter after — only an oracle that sees that "
               f"far ({lead['sharpe']:.2f}) starts to earn its fee.")
    if trad == "Fragile" and strict is not None and strict["dsharpe"] <= 0:
        one = one[:-1] + (f"; the lagged rule a real investor could run looks better on paper "
                          f"({real['sharpe']:.2f}) but loses to buy-and-hold with one more "
                          f"quarter of lag ({strict['sharpe']:.2f}).")
    return {"signal": signal, "signal_why": sig_why, "trad": trad, "trad_why": trad_why,
            "third": third, "third_why": third_why, "one_sentence": one}
