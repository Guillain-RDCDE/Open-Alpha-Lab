"""After-tax accounting, the break-even solver, and inference for Study 952.

The accounting, stated once and used everywhere
-----------------------------------------------
Every monthly total return is split into a **price** leg and an **income** leg
(``income = total_return - price_return``, see :func:`after_tax.data.decompose`). Only the
income leg is taxed each year:

- **muni** legs (MUB, VTEB, SUB, HYD) — coupon income is exempt from federal tax and from
  the 3.8% net-investment-income surtax. A *national* muni fund's income is, however,
  generally taxable by the holder's own state except for the in-state slice, so the muni
  income leg pays ``state_rate x (1 - muni_state_exempt_frac)``.
- **taxable credit** legs (AGG, LQD, VCIT) — coupon income pays ``fed + niit + state``.
- **cash** (BIL) — T-bill income pays ``fed + niit`` but is **state-exempt**.
- the **price** leg is left untaxed by default: a buy-and-hold holder's capital gain is
  unrealised. That is an **ASSUMPTION**, not tape; ``capgain_rate`` switches it on and
  :func:`capgain_sweep` shows what it costs.

Every rate here — the bracket, the surtax, the state rate, the in-state share, the
capital-gains treatment — is a **PROXY/ASSUMPTION**, not a measurement. The tape supplies
only the price and income legs. That is exactly why the headline of this study is a
*break-even bracket* rather than a single after-tax number: the break-even is the point at
which the assumption stops mattering.

The three questions
-------------------
1. **Break-even bracket.** At which effective marginal rate ``tau*`` do the after-tax mean
   returns of a muni leg and a taxable leg tie? Because the after-tax difference is exactly
   linear in ``tau``, ``tau*`` solves in closed form from two sample means — no search. The
   *identity* is exact; the *estimate* is not, and :func:`breakeven_ci` bootstraps it. On
   the real tape the total-return ``tau*`` is barely identified (MUB/VCIT: 35.0% point,
   95% CI ~[-11%, +82%]) because the price-leg difference it embeds is pure noise, while
   the **income-leg-only** break-even (:func:`income_breakeven`) is tight (~27% ± 3 pp).
   Quoting the first without the second, or either without its interval, is the single
   easiest way to over-sell this study.
2. **Is the after-tax win statistically real?** HAC (Newey-West) *t* on the monthly
   after-tax difference plus a circular block-bootstrap CI on its mean, an era cut, and an
   excess-of-*after-tax*-cash Sharpe race on both legs.
3. **Is it bankable?** As an asset-location choice (hold one *or* the other in a taxable
   account) the only friction is one entry: cost is one-way x NAV. As a long-short spread
   the short taxable leg pays **borrow**, swept in :func:`borrow_sweep`. The one arm with a
   signal — :func:`switch_overlay` — carries exactly **one execution lag**: the trailing
   12-month after-tax yields known through month ``t`` decide the position held in month
   ``t+1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12

# Statutory federal brackets swept in the headline, with the 3.8% net-investment-income
# (Medicare) surtax that applies above the MAGI thresholds. PROXY: real taxpayers face
# phase-outs, AMT and the muni de-minimis rule that these flat rates ignore.
NIIT = 0.038
BRACKETS = (
    ("0%", 0.00, 0.000),
    ("24%", 0.24, 0.000),
    ("24% + NIIT", 0.24, NIIT),
    ("32% + NIIT", 0.32, NIIT),
    ("37% + NIIT", 0.37, NIIT),
)

# Default classification of the desk tape.
KINDS = {
    "MUB": "muni", "VTEB": "muni", "SUB": "muni", "HYD": "muni",
    "AGG": "taxable", "LQD": "taxable", "VCIT": "taxable",
    "BIL": "cash",
    # synthetic column names
    "muni": "muni", "taxable": "taxable", "cash": "cash",
}


# --------------------------------------------------------------------------- #
# The tax profile
# --------------------------------------------------------------------------- #
def tax_profile(
    fed_rate: float = 0.37,
    niit: float = NIIT,
    state_rate: float = 0.0,
    muni_state_exempt_frac: float = 0.0,
    capgain_rate: float = 0.0,
) -> dict:
    """Bundle the (assumed) tax parameters. Every field is a PROXY, none is measured.

    ``muni_state_exempt_frac`` is the share of a national muni fund's income that is
    in-state (and so state-exempt); 0.0 is the conservative default for a national fund.
    ``capgain_rate`` taxes the price leg; 0.0 (the default) is the buy-and-hold,
    unrealised-gain assumption.
    """
    return {
        "fed_rate": float(fed_rate),
        "niit": float(niit),
        "state_rate": float(state_rate),
        "muni_state_exempt_frac": float(muni_state_exempt_frac),
        "capgain_rate": float(capgain_rate),
    }


def income_tax_rate(kind: str, profile: dict) -> float:
    """The marginal rate applied to one month of *income* for a leg of type ``kind``."""
    if kind == "muni":
        return profile["state_rate"] * (1.0 - profile["muni_state_exempt_frac"])
    if kind == "cash":  # T-bill interest: federal + surtax, state-exempt
        return profile["fed_rate"] + profile["niit"]
    return profile["fed_rate"] + profile["niit"] + profile["state_rate"]


def effective_rate(profile: dict) -> float:
    """The headline effective marginal rate on ordinary taxable interest."""
    return profile["fed_rate"] + profile["niit"] + profile["state_rate"]


def after_tax(panel: dict, column: str, profile: dict, kinds=None) -> pd.Series:
    """After-tax monthly simple return of one leg.

    ``after_tax = price x (1 - capgain_rate) + income x (1 - income_tax_rate)``. With the
    default ``capgain_rate = 0`` this is exactly ``total_return - tau x income``, so at a
    0% bracket the after-tax series is the total-return series, byte for byte.
    """
    kinds = KINDS if kinds is None else kinds
    kind = kinds[column]
    tau_i = income_tax_rate(kind, profile)
    tau_g = profile["capgain_rate"]
    price = panel["price"][column]
    income = panel["income"][column]
    return (price * (1.0 - tau_g) + income * (1.0 - tau_i)).rename(column)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def newey_west_t(x, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0. 6 monthly lags = half a year."""
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
    return float(mu / np.sqrt(var / n))


def one_sample_t(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def block_bootstrap_mean_ci(x, n_boot: int = 2000, block: int = 6,
                            seed: int = 952, alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the mean of a monthly series (bps/month).

    Six-month blocks preserve the heavy autocorrelation of bond-fund returns.
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"mean_bps": float("nan"), "ci_low_bps": float("nan"),
                "ci_high_bps": float("nan"), "frac_negative": float("nan"), "n_obs": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_bps": float(r.mean() * 1e4),
        "ci_low_bps": float(lo * 1e4),
        "ci_high_bps": float(hi * 1e4),
        "frac_negative": float((boots < 0).mean()),
        "n_obs": int(n), "block": int(block), "n_boot": int(n_boot),
    }


def summary(monthly: pd.Series) -> dict:
    """Annualised stats for a monthly simple-return series (pass an excess series for
    an excess Sharpe)."""
    r = pd.Series(monthly).astype(float).dropna()
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    sharpe = float(mu / sd * np.sqrt(MONTHS)) if sd > 0 else float("nan")
    wealth = (1.0 + r).cumprod()
    years = n / MONTHS
    cagr = (float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
            if years > 0 and wealth.iloc[-1] > 0 else float("nan"))
    dd = float((wealth / wealth.cummax() - 1.0).min())
    return {
        "n_months": int(n), "mean_bps": float(mu * 1e4), "ann_pct": float(mu * MONTHS * 100),
        "sharpe": sharpe, "vol_ann": float(sd * np.sqrt(MONTHS)),
        "cagr": cagr, "max_drawdown": dd, "tstat": newey_west_t(r.to_numpy()),
    }


# --------------------------------------------------------------------------- #
# The break-even bracket — the headline
# --------------------------------------------------------------------------- #
def breakeven_rate(panel: dict, muni: str, taxable: str,
                   profile: dict | None = None, kinds=None) -> dict:
    """The effective marginal rate at which the two legs' after-tax means tie.

    With the price leg untaxed, the after-tax difference in month *m* is

        d(tau) = [p_muni + i_muni (1 - s_m)] - [p_tax + i_tax (1 - tau)]
               = d(0 on the taxable side) + tau x i_tax

    which is **linear in tau**, so the break-even solves in closed form from two sample
    means: ``tau* = -mean(d_at_zero) / mean(i_tax)``. Interpretation matters:

    - ``0 < tau* < 1`` — a genuine tax-driven crossover: below ``tau*`` the taxable leg
      wins after tax, above it the muni leg does.
    - ``tau* <= 0`` — the muni leg already wins **pre-tax** on this sample. That is a
      duration/credit-composition result, *not* a tax result, and the pairing should not
      be read as evidence for the tax story.

    Scope caveat: ``tau*`` is the **federal + NIIT** increment solved on top of whatever
    ``profile`` already imposes. With the default (and every break-even reported in
    ``docs/results.md``) the state rate is 0, so ``tau*`` *is* the effective rate. Pass a
    profile carrying a state rate and the returned number is the federal piece only, on top
    of that state rate — not the headline effective rate.

    **Never quote this number without :func:`breakeven_ci`.** ``tau*`` is a ratio of two
    sample means whose numerator (the pre-tax total-return difference) is *not*
    statistically significant on the real tape, so the point estimate is far more precise-
    looking than the tape warrants: on the headline MUB/VCIT pairing the point estimate is
    35.0% and the 95% block-bootstrap interval is roughly [-11%, +82%]. Sweeping the tax
    assumptions leaves ``tau*`` untouched, which is a fact about the arithmetic (no tax
    parameter enters the numerator), **not** evidence that the estimate is precise.
    """
    profile = tax_profile(0.0, 0.0, 0.0) if profile is None else profile
    zero = dict(profile)
    zero["fed_rate"] = 0.0
    zero["niit"] = 0.0
    r_m = after_tax(panel, muni, zero, kinds)
    r_t = after_tax(panel, taxable, zero, kinds)
    idx = r_m.dropna().index.intersection(r_t.dropna().index)
    d0 = (r_m - r_t).reindex(idx)
    i_t = panel["income"][taxable].reindex(idx)
    mean_i = float(i_t.mean())
    tau = float("nan") if mean_i <= 0 else float(-d0.mean() / mean_i)
    return {
        "breakeven": tau,
        "pretax_diff_bps": float(d0.mean() * 1e4),
        "pretax_t": newey_west_t(d0.to_numpy()),
        "taxable_income_ann_pct": float(mean_i * MONTHS * 100),
        "muni_income_ann_pct": float(panel["income"][muni].reindex(idx).mean() * MONTHS * 100),
        "n_months": int(len(idx)),
        # POINT-ESTIMATE SIGN ONLY, not a test. The pre-tax difference is never
        # significant on this tape, so `tax_driven` says which side of zero the point
        # estimate fell on — see `breakeven_ci` for how wide that call actually is.
        "tax_driven": bool(np.isfinite(tau) and 0.0 < tau < 1.0),
    }


def breakeven_ci(panel: dict, muni: str, taxable: str, kinds=None, n_boot: int = 2000,
                 block: int = 6, seed: int = 952, alpha: float = 0.05,
                 top_bracket: float = 0.408) -> dict:
    """Circular block-bootstrap CI for the total-return break-even ``tau*``.

    **This is the honesty check on the study's headline number.** ``tau*`` is a *ratio of
    two sample means*, and its numerator — the pre-tax total-return difference — is never
    statistically significant on this tape. Sweeping the tax assumptions cannot reveal
    that, because the tax parameters do not enter the numerator: the noise lives in the
    **price** legs, which the sweeps never touch. Only resampling the tape does.

    Reported alongside the interval:

    - ``p_below_zero`` — bootstrap probability that the muni leg already wins **pre-tax**,
      i.e. that the "genuine tax-driven crossover" reading is the wrong side of zero.
    - ``p_above_top`` — bootstrap probability that ``tau*`` exceeds the top US statutory
      effective rate (37% + 3.8% NIIT = 40.8%), i.e. that **no** US bracket is high enough.
    - ``price_leg_diff_bps`` / ``price_leg_t`` — the price-leg difference that injects the
      noise, so the reader can see where the width comes from.
    """
    zero = tax_profile(0.0, 0.0, 0.0)
    r_m = after_tax(panel, muni, zero, kinds)
    r_t = after_tax(panel, taxable, zero, kinds)
    idx = r_m.dropna().index.intersection(r_t.dropna().index)
    d0 = (r_m - r_t).reindex(idx).to_numpy(dtype=float)
    i_t = panel["income"][taxable].reindex(idx).to_numpy(dtype=float)
    p_d = (panel["price"][muni] - panel["price"][taxable]).reindex(idx).to_numpy(dtype=float)
    n = len(idx)
    point = float("nan") if n == 0 or i_t.mean() <= 0 else float(-d0.mean() / i_t.mean())
    if n < block + 2:
        return {"breakeven": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_below_zero": float("nan"), "p_above_top": float("nan"), "n_months": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    taus = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        ii = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        mi = i_t[ii].mean()
        taus[b] = -d0[ii].mean() / mi if mi > 0 else np.nan
    taus = taus[np.isfinite(taus)]
    lo, hi = np.percentile(taus, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "breakeven": point, "ci_low": float(lo), "ci_high": float(hi),
        "p_below_zero": float((taus < 0).mean()),
        "p_above_top": float((taus > top_bracket).mean()),
        "price_leg_diff_bps": float(p_d.mean() * 1e4),
        "price_leg_t": newey_west_t(p_d),
        "n_months": int(n), "n_boot": int(len(taus)), "block": int(block),
    }


def tax_constant_decomposition(panel: dict, muni: str, taxable: str, profile: dict,
                               kinds=None) -> dict:
    """Split the after-tax difference into ``pre-tax difference`` + ``tax term``.

    **Why this exists.** The after-tax difference is ``d(tau) = d(0) + tau x i_taxable``.
    The second term is a *coupon stream*: it is large in the mean and almost constant in
    time (a bond fund's monthly distribution barely moves month to month), whereas ``d(0)``
    is a difference of two bond-fund total returns and is nearly all noise. Adding a
    near-deterministic constant to a noisy series lifts the mean without lifting the
    variance, so **the HAC t-stat rises mechanically with the bracket**. On this tape the
    tax term supplies ~85% of the MUB-vs-AGG mean and ~0.3% of its variance, which is the
    entire reason that pairing crosses ``|t| = 2`` at the top bracket and not at 0%.

    A ``|t| >= 2`` produced this way is **not** evidence of a market anomaly. It is a test
    of whether the tax code is nonzero, and it will pass by construction for any tax-exempt
    instrument against any taxable one. ``var_share_tax`` is the number that gives it away:
    when it is near zero, the significance is arithmetic, not empirical.
    """
    kinds = KINDS if kinds is None else kinds
    zero = tax_profile(0.0, 0.0, 0.0)
    r_m = after_tax(panel, muni, zero, kinds)
    r_t = after_tax(panel, taxable, zero, kinds)
    idx = r_m.dropna().index.intersection(r_t.dropna().index)
    d0 = (r_m - r_t).reindex(idx)
    tau_gap = income_tax_rate(kinds[taxable], profile) - income_tax_rate(kinds[muni], profile)
    tax_term = (tau_gap * panel["income"][taxable].reindex(idx)).rename("tax_term")
    total = (d0 + tax_term).rename("after_tax_diff")
    sd_tot = float(total.std(ddof=1))
    return {
        "n_months": int(len(idx)), "eff_rate": effective_rate(profile),
        "pretax_mean_bps": float(d0.mean() * 1e4), "pretax_sd_bps": float(d0.std(ddof=1) * 1e4),
        "pretax_t": newey_west_t(d0.to_numpy()),
        "tax_mean_bps": float(tax_term.mean() * 1e4),
        "tax_sd_bps": float(tax_term.std(ddof=1) * 1e4),
        "total_mean_bps": float(total.mean() * 1e4),
        "total_t": newey_west_t(total.to_numpy()),
        "mean_share_tax": (float(tax_term.mean() / total.mean())
                           if total.mean() != 0 else float("nan")),
        "var_share_tax": (float((tax_term.std(ddof=1) / sd_tot) ** 2)
                          if sd_tot > 0 else float("nan")),
    }


def income_breakeven(panel: dict, muni: str, taxable: str, kinds=None, n_boot: int = 2000,
                     block: int = 6, seed: int = 952, alpha: float = 0.05) -> dict:
    """The **income-leg-only** break-even: ``tau* = 1 - y_muni / y_taxable``.

    This is the brochure's tax-equivalent-yield rule, but computed on *realised, measured*
    monthly distribution yields (net of fund fees) instead of quoted yields — and it is the
    half of the break-even that this tape actually pins down, because the income legs are
    smooth. It is the correct number to quote when the question is "which coupon stream is
    worth more after tax". It is **not** the same question as the total-return break-even
    :func:`breakeven_rate` computes, which also asks the price legs to agree; on this tape
    the two answers differ by an amount that is pure noise, and reporting both is the whole
    honest content of the comparison.

    ``muni_state_exempt_frac`` / state tax are deliberately excluded here (federal + NIIT
    only) so this number is the like-for-like counterpart of the headline sweep at state 0.
    """
    ym = panel["income"][muni]
    yt = panel["income"][taxable]
    idx = ym.dropna().index.intersection(yt.dropna().index)
    a = ym.reindex(idx).to_numpy(dtype=float)
    b = yt.reindex(idx).to_numpy(dtype=float)
    n = len(idx)
    point = float("nan") if n == 0 or b.mean() <= 0 else float(1.0 - a.mean() / b.mean())
    if n < block + 2:
        return {"breakeven": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_months": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    taus = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        ii = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        mb = b[ii].mean()
        taus[k] = 1.0 - a[ii].mean() / mb if mb > 0 else np.nan
    taus = taus[np.isfinite(taus)]
    lo, hi = np.percentile(taus, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "breakeven": point, "ci_low": float(lo), "ci_high": float(hi),
        "muni_income_ann_pct": float(a.mean() * MONTHS * 100),
        "taxable_income_ann_pct": float(b.mean() * MONTHS * 100),
        "n_months": int(n), "n_boot": int(len(taus)), "block": int(block),
    }


# --------------------------------------------------------------------------- #
# The after-tax race
# --------------------------------------------------------------------------- #
def race(panel: dict, muni: str, taxable: str, profile: dict,
         cash: str = "BIL", cost_bps: float = 3.0, kinds=None,
         months: pd.Index | None = None) -> dict:
    """Race one muni leg against one taxable leg **after tax**, excess of after-tax cash.

    The framing is **asset location**: a taxable-account holder owns one leg or the other,
    so the only friction on the *switch* is a single round trip (sell the incumbent, buy
    the challenger) borne by the muni arm — ``2 x cost_bps`` one-way x NAV, amortised
    across the sample. The incumbent taxable arm pays nothing. Fund expense ratios are
    already inside the total-return tape and are not charged again. There is no short leg
    here, so no borrow — see :func:`borrow_sweep` for the long-short spread framing, where
    the short taxable leg does pay borrow.

    Sharpes are reported **gross** and **net** of that switch cost, both excess of the
    *after-tax* cash leg. When the taxable leg *is* the cash leg (SUB vs BIL) there is no
    third asset to subtract, so the Sharpes are reported on the raw after-tax series and
    ``excess_of_cash`` is False.
    """
    kinds = KINDS if kinds is None else kinds
    r_m = after_tax(panel, muni, profile, kinds).dropna()
    r_t = after_tax(panel, taxable, profile, kinds).dropna()
    idx = r_m.index.intersection(r_t.index)
    excess_of_cash = cash in panel["total"].columns and cash not in (muni, taxable)
    if excess_of_cash:
        r_c = after_tax(panel, cash, profile, kinds).dropna()
        idx = idx.intersection(r_c.index)
    else:
        r_c = pd.Series(0.0, index=idx)
    if months is not None:
        idx = idx.intersection(months)
    r_m, r_t, r_c = r_m.reindex(idx), r_t.reindex(idx), r_c.reindex(idx)

    n = max(len(idx), 1)
    switch_drag = (2.0 * cost_bps * 1e-4) / n   # one round trip, amortised, muni arm only
    diff = (r_m - switch_drag - r_t).rename("after_tax_diff")
    e_m_gross = (r_m - r_c).rename("excess_muni_gross")
    e_m = (r_m - switch_drag - r_c).rename("excess_muni")
    e_t = (r_t - r_c).rename("excess_taxable")

    s_mg, s_m, s_t = summary(e_m_gross), summary(e_m), summary(e_t)
    boot = block_bootstrap_mean_ci(diff)
    return {
        "muni": muni, "taxable": taxable, "n_months": int(len(idx)),
        "eff_rate": effective_rate(profile), "excess_of_cash": excess_of_cash,
        "diff_bps": float(diff.mean() * 1e4),
        "diff_ann_pct": float(diff.mean() * MONTHS * 100),
        "t_diff": newey_west_t(diff.to_numpy()),
        "ci_low_bps": boot["ci_low_bps"], "ci_high_bps": boot["ci_high_bps"],
        "sharpe_muni_gross": s_mg["sharpe"], "sharpe_muni": s_m["sharpe"],
        "sharpe_taxable": s_t["sharpe"],
        "sharpe_edge": s_m["sharpe"] - s_t["sharpe"],
        "cagr_muni": s_m["cagr"], "cagr_taxable": s_t["cagr"],
        "dd_muni": s_m["max_drawdown"], "dd_taxable": s_t["max_drawdown"],
        "vol_muni": s_m["vol_ann"], "vol_taxable": s_t["vol_ann"],
        "diff": diff, "excess_muni": e_m, "excess_taxable": e_t,
    }


def bracket_sweep(panel: dict, muni: str, taxable: str, state_rate: float = 0.0,
                  brackets=BRACKETS, kinds=None, **kw) -> pd.DataFrame:
    """The after-tax difference and its HAC *t* across the statutory bracket ladder."""
    rows = []
    for label, fed, niit in brackets:
        prof = tax_profile(fed_rate=fed, niit=niit, state_rate=state_rate)
        r = race(panel, muni, taxable, prof, kinds=kinds, **kw)
        rows.append({
            "bracket": label, "eff_rate": r["eff_rate"], "diff_bps": r["diff_bps"],
            "diff_ann_pct": r["diff_ann_pct"], "t_diff": r["t_diff"],
            "sharpe_muni": r["sharpe_muni"], "sharpe_taxable": r["sharpe_taxable"],
        })
    return pd.DataFrame(rows)


def state_sweep(panel: dict, muni: str, taxable: str,
                state_rates=(0.0, 0.05, 0.093, 0.133),
                in_state_fracs=(0.0, 1.0), fed_rate: float = 0.37,
                kinds=None, **kw) -> pd.DataFrame:
    """Sweep the state-tax ASSUMPTION: the rate, and the in-state (exempt) income share."""
    rows = []
    for s in state_rates:
        for f in in_state_fracs:
            prof = tax_profile(fed_rate=fed_rate, state_rate=s, muni_state_exempt_frac=f)
            r = race(panel, muni, taxable, prof, kinds=kinds, **kw)
            rows.append({"state_rate": s, "in_state_frac": f, "eff_rate": r["eff_rate"],
                         "diff_bps": r["diff_bps"], "t_diff": r["t_diff"]})
    return pd.DataFrame(rows)


def capgain_sweep(panel: dict, muni: str, taxable: str,
                  capgain_rates=(0.0, 0.15, 0.238), fed_rate: float = 0.37,
                  kinds=None, **kw) -> pd.DataFrame:
    """Sweep the buy-and-hold ASSUMPTION that the price leg is never realised."""
    rows = []
    for g in capgain_rates:
        prof = tax_profile(fed_rate=fed_rate, capgain_rate=g)
        r = race(panel, muni, taxable, prof, kinds=kinds, **kw)
        rows.append({"capgain_rate": g, "diff_bps": r["diff_bps"], "t_diff": r["t_diff"]})
    return pd.DataFrame(rows)


def cost_sweep(panel: dict, muni: str, taxable: str, fed_rate: float = 0.37,
               cost_bps_grid=(0.0, 3.0, 10.0, 25.0), kinds=None, **kw) -> pd.DataFrame:
    """One-way entry cost x NAV, gross to net, on the asset-location choice."""
    rows = []
    prof = tax_profile(fed_rate=fed_rate)
    for c in cost_bps_grid:
        r = race(panel, muni, taxable, prof, cost_bps=c, kinds=kinds, **kw)
        rows.append({"cost_bps": c, "diff_bps": r["diff_bps"], "t_diff": r["t_diff"],
                     "sharpe_muni": r["sharpe_muni"], "sharpe_taxable": r["sharpe_taxable"]})
    return pd.DataFrame(rows)


def borrow_sweep(panel: dict, muni: str, taxable: str, fed_rate: float = 0.37,
                 borrow_bps_yr=(0, 25, 50, 100), kinds=None, **kw) -> pd.DataFrame:
    """The long-muni / short-taxable **spread**, where the short leg pays borrow.

    Borrow is charged as ``borrow_bps_yr / 12`` against NAV every month the short is on
    (always, in this always-on spread). This is the framing under which the trade is a
    trade rather than an asset-location choice — and it is where the thin edge dies.
    """
    prof = tax_profile(fed_rate=fed_rate)
    base = race(panel, muni, taxable, prof, kinds=kinds, **kw)
    rows = []
    for b in borrow_bps_yr:
        drag = (b * 1e-4) / MONTHS
        d = base["diff"] - drag
        rows.append({"borrow_bps_yr": b, "diff_bps": float(d.mean() * 1e4),
                     "t_diff": newey_west_t(d.to_numpy()),
                     "ann_pct": float(d.mean() * MONTHS * 100)})
    return pd.DataFrame(rows)


def era_cut(panel: dict, muni: str, taxable: str, profile: dict,
            split: str = "2017-01", kinds=None, **kw) -> dict:
    """Split the monthly sample at ``split`` and re-run the after-tax race on each half."""
    idx = panel["total"].index
    cut = pd.Period(split, freq="M")
    out = {}
    for tag, mask in [("early", idx < cut), ("late", idx >= cut)]:
        months = idx[mask]
        if len(months) < 24:
            out[tag] = None
            continue
        r = race(panel, muni, taxable, profile, kinds=kinds, months=months, **kw)
        out[tag] = {k: r[k] for k in ("n_months", "diff_bps", "diff_ann_pct", "t_diff",
                                      "sharpe_muni", "sharpe_taxable")}
    return out


def income_floor_sensitivity(total_daily, price_daily, muni: str, taxable: str,
                             fed_rate: float = 0.37, kinds=None) -> pd.DataFrame:
    """Re-run the headline with the income floor on and off (the reconstruction choice)."""
    from . import data as _data

    prof = tax_profile(fed_rate=fed_rate)
    raw = _data.decompose(total_daily, price_daily, floor_income=False)["income"][taxable]
    neg = int((raw < 0).sum())
    clipped_bps = float(-raw.clip(upper=0.0).mean() * 1e4)  # mean amount the floor adds
    rows = []
    for floor in (True, False):
        panel = _data.decompose(total_daily, price_daily, floor_income=floor)
        r = race(panel, muni, taxable, prof, kinds=kinds)
        be = breakeven_rate(panel, muni, taxable, kinds=kinds)
        rows.append({"floor_income": floor, "diff_bps": r["diff_bps"],
                     "t_diff": r["t_diff"], "breakeven": be["breakeven"],
                     "negative_income_months": neg,
                     "floor_adds_bps_per_month": clipped_bps})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The one arm with a signal — and therefore exactly one execution lag
# --------------------------------------------------------------------------- #
def switch_overlay(panel: dict, muni: str, taxable: str, profile: dict,
                   lookback: int = 12, cost_bps: float = 3.0, kinds=None) -> dict:
    """Hold whichever leg carried the higher trailing after-tax income yield.

    The trailing ``lookback``-month after-tax income yields are known at the close of
    month ``t``; the position they imply is held through month ``t+1``. That ``shift(1)``
    is the study's **single execution lag**. Each switch pays ``cost_bps`` one-way x NAV.
    A rule that cannot beat simply holding the muni leg tells you the bracket question has
    no timing dimension worth trading.
    """
    kinds = KINDS if kinds is None else kinds
    tau_m = income_tax_rate(kinds[muni], profile)
    tau_t = income_tax_rate(kinds[taxable], profile)
    y_m = panel["income"][muni].rolling(lookback, min_periods=lookback).sum() * (1 - tau_m)
    y_t = panel["income"][taxable].rolling(lookback, min_periods=lookback).sum() * (1 - tau_t)

    hold_muni = (y_m > y_t).astype(float).shift(1)  # <- the one execution lag
    r_m = after_tax(panel, muni, profile, kinds)
    r_t = after_tax(panel, taxable, profile, kinds)
    idx = hold_muni.dropna().index.intersection(r_m.dropna().index).intersection(r_t.dropna().index)
    sig = hold_muni.reindex(idx)
    switches = sig.diff().abs().fillna(0.0)
    gross = sig * r_m.reindex(idx) + (1 - sig) * r_t.reindex(idx)
    net = gross - switches * (cost_bps * 1e-4)
    bench = r_m.reindex(idx)
    d = (net - bench).rename("overlay_minus_muni")
    return {
        "n_months": int(len(idx)), "in_muni_frac": float(sig.mean()),
        "n_switches": int(switches.sum()),
        "overlay_ann_pct": float(net.mean() * MONTHS * 100),
        "muni_ann_pct": float(bench.mean() * MONTHS * 100),
        "edge_bps": float(d.mean() * 1e4), "t_edge": newey_west_t(d.to_numpy()),
        "sharpe_overlay": summary(net)["sharpe"], "sharpe_muni": summary(bench)["sharpe"],
        "signal": sig,
    }


# --------------------------------------------------------------------------- #
# Synthetic control — machinery proof only; never supports a real-tape stamp
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict, fed_rate: float = 0.37, kinds=None) -> dict:
    """Run the break-even solver and the after-tax race on a synthetic panel.

    On the planted world (``signal_strength=1``) the solver must recover the *theoretical*
    break-even implied by the planted coupon yields, and the after-tax difference must
    flip sign as the bracket crosses it. On the null (``signal_strength=0``, statistical
    twins) the break-even must collapse to ~0 and the **pre-tax** difference must be
    indistinguishable from zero.

    Note what the null does *not* claim: at a positive bracket the after-tax difference on
    the null world is large and highly significant — because it is pure tax arithmetic
    (``tau x yield``), which is exactly the mechanical effect this study is measuring. The
    null's job is to prove the machinery attributes none of it to a market edge.
    """
    kinds = KINDS if kinds is None else kinds
    be = breakeven_rate(panel, "muni", "taxable", kinds=kinds)
    lo = race(panel, "muni", "taxable", tax_profile(0.0, 0.0, 0.0), cash="cash",
              cost_bps=0.0, kinds=kinds)
    hi = race(panel, "muni", "taxable", tax_profile(fed_rate, NIIT, 0.0), cash="cash",
              cost_bps=0.0, kinds=kinds)
    below = race(panel, "muni", "taxable",
                 tax_profile(max(be["breakeven"] - 0.15, 0.0), 0.0, 0.0), cash="cash",
                 cost_bps=0.0, kinds=kinds)
    above = race(panel, "muni", "taxable",
                 tax_profile(be["breakeven"] + 0.15, 0.0, 0.0), cash="cash",
                 cost_bps=0.0, kinds=kinds)
    return {
        "breakeven": be["breakeven"], "tax_driven": be["tax_driven"],
        "pretax_diff_bps": be["pretax_diff_bps"], "pretax_t": be["pretax_t"],
        "diff_bps_at_zero": lo["diff_bps"], "t_at_zero": lo["t_diff"],
        "diff_bps_at_top": hi["diff_bps"], "t_at_top": hi["t_diff"],
        "diff_below_breakeven": below["diff_bps"], "diff_above_breakeven": above["diff_bps"],
        "n_months": hi["n_months"],
    }
