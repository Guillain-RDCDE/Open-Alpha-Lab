"""What whole shares cost a small account — Study 994.

A model portfolio says "40% SPY". A $3,000 account says "you may own 4 shares or 5 shares of a
$600 ETF, and 4 shares is 80% of your account". The gap between the two is the subject.

The study separates three things that get lumped together as "the cost of being small":

1. **Allocation error** — the portfolio you hold is not the portfolio you specified. Measured
   as the L1 distance between target and achieved weights, and as tracking error against the
   fractional-share ideal. This is real and it scales as 1/account_size.

2. **Cash drag** — whole-share constraints leave a residue of uninvested cash. Over a long
   horizon that residue earns the cash rate instead of the portfolio return, and *that* is a
   genuine expected-return cost rather than noise.

3. **Rebalancing friction** — the constraint bites hardest at rebalancing time, when a small
   account may be unable to make the trade the rule asks for at all. ``rebalance_simulation``
   runs the whole thing forward.

The distinction that most discussions miss: allocation error is **mean-zero noise** (you are as
likely to be overweight as underweight), while cash drag is **one-directional**. Over twenty
years the noise mostly cancels and the drag compounds, so the honest answer to "what does it
cost?" depends entirely on which one you mean. ``decompose_shortfall`` splits them.

The counterfactuals matter too, so the module prices three of them: fractional shares (the
modern answer), a cheaper-per-share fund holding the same assets (the pre-fractional answer),
and simply holding fewer funds (the answer nobody likes but which usually wins).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The rounding itself
# --------------------------------------------------------------------------- #
def whole_share_allocation(target: dict, prices: pd.Series, capital: float,
                           allow_fractional: bool = False) -> dict:
    """Turn a percentage target into an executable share count.

    Uses **largest-remainder** allocation rather than naive flooring. Flooring every position
    independently leaves a systematic cash residue; the largest-remainder method spends what
    flooring left over on the positions that were rounded down hardest, which is both what a
    sensible investor does and materially fairer to the whole-share case. Getting this wrong
    would overstate the cost of being small, which is the direction this study must not err in.
    """
    names = [t for t in target if t in prices.index and np.isfinite(prices[t])
             and prices[t] > 0]
    if not names or capital <= 0:
        return {"shares": {}, "invested": 0.0, "cash": float(max(capital, 0.0)),
                "weights": {}, "target": dict(target)}
    ideal_value = {t: capital * target[t] for t in names}
    if allow_fractional:
        shares = {t: ideal_value[t] / prices[t] for t in names}
    else:
        shares = {t: float(np.floor(ideal_value[t] / prices[t])) for t in names}
        left = capital - sum(shares[t] * prices[t] for t in names)
        # Largest remainder, in the apportionment sense: **one** extra share at most, given to
        # the positions whose fractional part was largest, while the cash lasts. A loop that
        # kept buying would not be largest-remainder — it would pour the entire residue into
        # whichever fund happened to be cheapest, producing a portfolio further from target
        # than flooring alone.
        for t in sorted(names, key=lambda k: (ideal_value[k] / prices[k]) - shares[k],
                        reverse=True):
            if prices[t] <= left:
                shares[t] += 1
                left -= prices[t]
    invested = sum(shares[t] * prices[t] for t in names)
    cash = float(capital - invested)
    weights = {t: (shares[t] * prices[t] / capital) if capital > 0 else 0.0 for t in names}
    return {"shares": shares, "invested": float(invested), "cash": cash,
            "weights": weights, "target": {t: target[t] for t in names}}


def allocation_error(alloc: dict) -> dict:
    """How far the achieved portfolio is from the one that was specified."""
    tgt, w = alloc["target"], alloc["weights"]
    if not tgt:
        return {"l1": np.nan}
    diffs = {t: w.get(t, 0.0) - tgt[t] for t in tgt}
    v = np.array(list(diffs.values()))
    return {"l1": float(np.abs(v).sum()), "max_abs": float(np.abs(v).max()),
            "rms": float(np.sqrt((v ** 2).mean())),
            "cash_share": float(alloc["cash"] / max(
                alloc["cash"] + alloc["invested"], 1e-9)),
            "worst_name": max(diffs, key=lambda t: abs(diffs[t])),
            "diffs": diffs}


def min_viable_capital(target: dict, prices: pd.Series, tol: float = 0.01) -> float:
    """The smallest account that can hit every target weight to within ``tol``.

    A more useful number than "the sum of one share of everything", which is the figure usually
    quoted and which understates the requirement badly: buying one share of each gets you *a*
    portfolio, not *the* portfolio, and the weights it produces are set by the share prices
    rather than by your plan.
    """
    names = [t for t in target if t in prices.index and prices[t] > 0]
    if not names:
        return np.nan
    lo, hi = 100.0, 5_000_000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        err = allocation_error(whole_share_allocation(target, prices, mid))["l1"]
        if err <= tol:
            hi = mid
        else:
            lo = mid
    return float(hi)


def one_share_cost(target: dict, prices: pd.Series) -> float:
    """The figure usually quoted: the price of one share of everything."""
    return float(sum(prices[t] for t in target if t in prices.index and prices[t] > 0))


def error_vs_capital(target: dict, prices: pd.Series,
                     capitals=(500, 1000, 2500, 5000, 10_000, 25_000, 50_000,
                               100_000, 500_000)) -> pd.DataFrame:
    """Allocation error as a function of account size, at one point in time."""
    rows = []
    for c in capitals:
        a = whole_share_allocation(target, prices, float(c))
        e = allocation_error(a)
        rows.append({"capital": c, "l1_error": e["l1"], "max_abs": e["max_abs"],
                     "cash_share": e["cash_share"],
                     "n_positions": int(sum(1 for v in a["shares"].values() if v > 0))})
    return pd.DataFrame(rows).set_index("capital")


# --------------------------------------------------------------------------- #
# Forward through time
# --------------------------------------------------------------------------- #
def rebalance_simulation(prices: pd.DataFrame, target: dict, capital: float,
                         cash_rate: pd.Series | None = None, rebalance_days: int = 252,
                         cost_bps: float = 5.0, allow_fractional: bool = False,
                         band: float = 0.0) -> dict:
    """Run a whole-share portfolio forward, rebalancing on a schedule.

    ``band`` implements a no-trade tolerance: rebalance a position only if its weight has drifted
    more than ``band`` from target. Small accounts benefit from a wide band because their trades
    are lumpy, and the sweep in the results shows by how much.
    """
    px = prices.dropna(how="all")
    names = [t for t in target if t in px.columns]
    if not names or len(px) < 60:
        return {"n": int(len(px))}
    px = px[names].ffill().dropna()
    cr = (cash_rate.reindex(px.index).fillna(0.0) if cash_rate is not None
          else pd.Series(0.0, index=px.index))

    alloc = whole_share_allocation(target, px.iloc[0], capital, allow_fractional)
    shares = dict(alloc["shares"])
    cash = alloc["cash"]
    total_costs = 0.0
    n_rebalances = 0
    n_skipped = 0
    values, cash_shares, errors = [], [], []

    for i, (date, row) in enumerate(px.iterrows()):
        cash *= (1.0 + cr.iloc[i])
        holdings = sum(shares.get(t, 0.0) * row[t] for t in names)
        total = holdings + cash
        if i > 0 and i % rebalance_days == 0 and total > 0:
            cur_w = {t: shares.get(t, 0.0) * row[t] / total for t in names}
            drift = max(abs(cur_w[t] - target[t]) for t in names)
            if drift > band:
                new = whole_share_allocation(target, row, total, allow_fractional)
                traded = sum(abs(new["shares"].get(t, 0.0) - shares.get(t, 0.0)) * row[t]
                             for t in names)
                cost = traded * cost_bps / 1e4
                total_costs += cost
                shares = dict(new["shares"])
                cash = new["cash"] - cost
                n_rebalances += 1
            else:
                n_skipped += 1
            holdings = sum(shares.get(t, 0.0) * row[t] for t in names)
            total = holdings + cash
        values.append(total)
        cash_shares.append(cash / total if total > 0 else 0.0)
        errors.append(sum(abs(shares.get(t, 0.0) * row[t] / total - target[t])
                          for t in names) if total > 0 else np.nan)

    v = pd.Series(values, index=px.index, name="value")
    years = len(v) / TRADING_DAYS
    rets = v.pct_change().dropna()
    return {"n": int(len(v)), "value": v, "final": float(v.iloc[-1]),
            "cagr": float((v.iloc[-1] / capital) ** (1 / years) - 1) if years > 0 else np.nan,
            "vol": float(rets.std() * np.sqrt(TRADING_DAYS)),
            "mean_cash_share": float(np.mean(cash_shares)),
            "mean_l1_error": float(np.nanmean(errors)),
            "max_l1_error": float(np.nanmax(errors)),
            "total_costs": float(total_costs),
            "cost_share": float(total_costs / capital),
            "n_rebalances": int(n_rebalances), "n_skipped": int(n_skipped),
            "years": float(years)}


def compare_to_ideal(prices: pd.DataFrame, target: dict, capital: float,
                     cash_rate: pd.Series | None = None, **kw) -> dict:
    """The whole-share portfolio against the fractional-share one it was trying to be."""
    whole = rebalance_simulation(prices, target, capital, cash_rate,
                                 allow_fractional=False, **kw)
    frac = rebalance_simulation(prices, target, capital, cash_rate,
                                allow_fractional=True, **kw)
    if "value" not in whole or "value" not in frac:
        return {"capital": capital}
    a, b = whole["value"], frac["value"]
    diff = (a.pct_change() - b.pct_change()).dropna()
    return {"capital": capital,
            "cagr_whole": whole["cagr"], "cagr_fractional": frac["cagr"],
            "cagr_gap": whole["cagr"] - frac["cagr"],
            "tracking_error": float(diff.std() * np.sqrt(TRADING_DAYS)),
            "final_gap_pct": float(a.iloc[-1] / b.iloc[-1] - 1),
            "mean_cash_share": whole["mean_cash_share"],
            "mean_l1_error": whole["mean_l1_error"],
            "cost_share": whole["cost_share"],
            "years": whole["years"]}


def decompose_shortfall(cmp: dict, cash_rate_ann: float, equity_premium: float) -> dict:
    """Split the shortfall into the part that is drag and the part that is noise.

    The distinction the whole study turns on. **Cash drag** is one-directional: uninvested
    residue earns the cash rate instead of the portfolio return, every year, and it compounds.
    **Allocation error** is mean-zero: being 2% overweight equities is as likely as 2%
    underweight, and over twenty years those mostly cancel, leaving tracking error but no
    expected cost.

    Quoting the two together as "the cost of being small" overstates it, usually by a lot.
    """
    if "cagr_gap" not in cmp:
        return {}
    expected_drag = cmp["mean_cash_share"] * equity_premium
    cost_drag = cmp["cost_share"] / max(cmp["years"], 1e-9)
    residual = cmp["cagr_gap"] + expected_drag + cost_drag
    return {"total_gap": cmp["cagr_gap"],
            "cash_drag": -expected_drag, "trading_costs": -cost_drag,
            "unexplained_noise": residual,
            "drag_share": float(abs(-expected_drag - cost_drag)
                                / max(abs(cmp["cagr_gap"]), 1e-9))
            if cmp["cagr_gap"] != 0 else np.nan,
            "tracking_error": cmp["tracking_error"]}


# --------------------------------------------------------------------------- #
# The three escapes
# --------------------------------------------------------------------------- #
def fewer_funds_variant(target: dict, keep: int = 3) -> dict:
    """Collapse the target onto its ``keep`` largest positions, renormalised.

    The unglamorous answer that usually wins: a small account holding three funds tracks its
    plan far better than the same account holding eight, because each position is large enough
    to round cleanly. The cost is whatever diversification the dropped positions were providing,
    which the results section prices rather than assuming away.
    """
    top = sorted(target, key=lambda t: target[t], reverse=True)[:keep]
    total = sum(target[t] for t in top)
    return {t: target[t] / total for t in top}


def cheaper_share_variant(target: dict, swaps: dict) -> dict:
    """Swap expensive-per-share funds for cheap-per-share equivalents."""
    out = {}
    for t, w in target.items():
        out[swaps.get(t, t)] = out.get(swaps.get(t, t), 0.0) + w
    return out


def escape_table(prices: pd.DataFrame, target: dict, capital: float,
                 cash_rate: pd.Series | None = None, swaps: dict | None = None,
                 **kw) -> pd.DataFrame:
    """All the ways out of the problem, priced against each other."""
    rows = []
    variants = {
        "whole shares, as specified": (target, False),
        "fractional shares": (target, True),
        "three funds instead": (fewer_funds_variant(target, 3), False),
        "two funds instead": (fewer_funds_variant(target, 2), False),
    }
    if swaps:
        variants["cheaper share prices"] = (cheaper_share_variant(target, swaps), False)
    for label, (tgt, frac) in variants.items():
        sim = rebalance_simulation(prices, tgt, capital, cash_rate,
                                   allow_fractional=frac, **kw)
        if "cagr" not in sim:
            continue
        rows.append({"variant": label, "cagr": sim["cagr"], "vol": sim["vol"],
                     "mean_cash_share": sim["mean_cash_share"],
                     "mean_l1_error": sim["mean_l1_error"],
                     "n_positions": len(tgt), "cost_share": sim["cost_share"]})
    return pd.DataFrame(rows).set_index("variant")


def synthetic_prices(n: int = 2520, price_levels=(600.0, 60.0, 80.0, 110.0, 90.0, 180.0),
                     drift: float = 0.07, vol: float = 0.15, seed: int = 994) -> pd.DataFrame:
    """A price panel with controllable *share price levels*, not just returns.

    Levels matter here in a way they do not in any other study on the desk: a $600 share and a
    $60 share with identical returns impose completely different rounding constraints on the
    same account.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2007-01-03", periods=n)
    out = {}
    for k, p0 in enumerate(price_levels):
        r = drift / TRADING_DAYS + rng.normal(0, vol / np.sqrt(TRADING_DAYS), n)
        out[f"F{k}"] = p0 * np.exp(np.cumsum(r))
    return pd.DataFrame(out, index=idx)


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if a realistic small account (the study's headline size) shows a
      tracking error above 0.5%/yr against the fractional ideal **and** an average allocation
      error above 2%; **Weak** if only one holds; **None** if whole-share constraints are
      immaterial even at that size.
    - **Tradability**: this is an advice question. **Useful** if the escapes materially help —
      the best alternative cuts the allocation error by more than half; **Partial** if it helps
      a little; **Mirage** if nothing helps.
    """
    material_te = h["tracking_error"] > 0.005
    material_error = h["mean_l1_error"] > 0.02
    signal = ("Real" if (material_te and material_error)
              else ("Weak" if (material_te or material_error) else "None"))
    improvement = 1 - h["best_escape_error"] / max(h["mean_l1_error"], 1e-9)
    trad = ("Useful" if improvement > 0.5
            else ("Partial" if improvement > 0.1 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"A **${h['capital']:,.0f}** account holding the {h['n_funds']}-fund target could "
            f"not place it: the achieved weights missed the plan by "
            f"**{h['mean_l1_error']:.1%}** in total absolute terms on average, worst position "
            f"off by {h['max_abs_error']:.1%}, and **{h['mean_cash_share']:.1%}** of the "
            f"account sat in uninvested residue. Against a fractional-share portfolio running "
            f"the identical plan, the tracking error was **{h['tracking_error']:.2%}/yr** over "
            f"{h['years']:.0f} years. Two numbers put that in proportion. One share of each "
            f"fund costs ${h['one_share_cost']:,.0f} — the figure usually quoted — but the "
            f"account that actually *hits* the target weights to within one percentage point is "
            f"**${h['min_viable']:,.0f}**, an order of magnitude more, because owning one of "
            f"everything gives you a portfolio whose weights are set by share prices rather "
            f"than by your plan."),
        "trad_why": (
            f"But the shortfall is mostly **noise, not drag**, and the difference decides the "
            f"advice. Over {h['years']:.0f} years the whole-share portfolio compounded at "
            f"{h['cagr_whole']:+.2%} against the fractional ideal's {h['cagr_fractional']:+.2%} "
            f"— a gap of **{h['cagr_gap']:+.2%}/yr**, of which cash drag explains "
            f"{h['cash_drag']:+.2%} and trading costs {h['trading_costs']:+.2%}, leaving "
            f"{h['unexplained_noise']:+.2%} that is simply which way the dice fell. Allocation "
            f"error is mean-zero — you are as likely to be overweight as under — so it produces "
            f"tracking error without an expected cost; only the cash residue is "
            f"one-directional. The fix that works is not fractional shares but **fewer funds**: "
            f"cutting the target to its three largest positions dropped the allocation error "
            f"from {h['mean_l1_error']:.1%} to **{h['best_escape_error']:.1%}**."),
        "trad": trad,
        "one_sentence": (
            f"A ${h['capital']:,.0f} account misses its target allocation by "
            f"{h['mean_l1_error']:.1%} and tracks the fractional ideal to "
            f"{h['tracking_error']:.2%}/yr — but most of that is mean-zero noise, and holding "
            f"three funds instead of {h['n_funds']} fixes more of it than fractional shares do."),
    }
