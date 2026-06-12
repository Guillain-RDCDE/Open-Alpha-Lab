"""The falsification battery — does the Markov machine add anything the price doesn't already have?

Five questions, each pre-registered in the README's beat 3:

  1. **Headline.** Across many markets, does the machine's signed bet capture a realized edge
     over the price — i.e. is ``sign(edge) * (outcome - price)`` positive with HAC *t* > 2?
     On efficient (martingale) markets it must be indistinguishable from zero. → ``headline_test``.
  2. **Noise, not signal.** Is the raw Monte-Carlo "edge" a zero-mean term whose size *shrinks
     as history grows* (the signature of estimation noise, not information)? → ``edge_vs_history``.
  3. **Inertness.** Are the trade decisions essentially unchanged if you delete the whole
     Markov + Monte-Carlo stage (``raw_prob := price``)? And is the "system edge" just the
     calibration curve ``calibrate(price) - price`` regardless of the chain? → ``inertness``.
  4. **Tradability.** Kelly-size the machine's bets and score them against the *true* outcome,
     net of a realistic bid/ask. Bankroll trajectory, per-trade Sharpe (bootstrap CI), win
     rate. → ``pnl_sim`` / ``cost_sweep``.
  5. **Machinery sanity / could-you-trade-it.** On markets with a *planted* favorite-longshot
     wedge, recover the edge with an oracle price→truth lookup — both cost-blind (the gross
     ceiling) and cost-aware (the honest net reference: trade only where the wedge clears the
     entry toll) — show where it lives, and ask whether the chain adds anything once a real
     edge is present. → ``recover_planted``.

Inference and bootstrap come from the shared engine (`quantlab.analytics`, `quantlab.stats`)
so the numbers line up with every other study on the desk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.analytics import mean_tstat_hac
from quantlab.stats import sharpe_ci_bootstrap

from . import data
from .markov import CALIBRATION, MarkovMintSystem, calibrate


# --------------------------------------------------------------------------- #
# Run the machine across a basket of markets
# --------------------------------------------------------------------------- #

def analyze_markets(markets, system: MarkovMintSystem | None = None,
                    ablate_markov: bool = False, seed: int = 0) -> pd.DataFrame:
    """Run the article's pipeline on every market; one row per market.

    A single seeded RNG drives every market's Monte-Carlo so a re-run is bit-for-bit
    reproducible. Columns include the raw/calibrated probabilities, the raw and system edges,
    the signed direction, the Kelly fraction, and the ground-truth ``outcome``/``true_prob``.
    """
    system = system or MarkovMintSystem()
    rng = np.random.default_rng(seed)
    rows = []
    for m in markets:
        res = system.analyze(m.prices, m.current_price, m.days_to_expiry,
                             rng=rng, ablate_markov=ablate_markov)
        res["outcome"] = m.outcome
        res["true_prob"] = m.true_prob
        rows.append(res)
    df = pd.DataFrame(rows)
    df["dir_sign"] = np.where(df["direction"] == "BUY YES", 1.0,
                              np.where(df["direction"] == "BUY NO", -1.0, 0.0))
    return df


# --------------------------------------------------------------------------- #
# 1. Headline: does the signed bet capture realized edge over the price?
# --------------------------------------------------------------------------- #

def headline_test(df: pd.DataFrame) -> dict:
    """Realized directional edge per *active* trade, HAC-tested, vs an oracle benchmark.

    For every non-PASS market the machine takes a side; the edge it actually captures over the
    price is ``dir_sign * (outcome - price)`` (a YES bet wins ``1-price`` when the event
    happens, loses ``price`` when it doesn't — its mean is the true mispricing it found). We
    HAC-*t* that mean. The oracle column does the same using ``sign(true_prob - price)`` over
    *all* markets — the most any method could capture given the true probabilities. On an
    efficient market both are zero; the gap between them is what the chain leaves on the table.
    """
    active = df[df["dir_sign"] != 0.0]
    realized = (active["dir_sign"] * (active["outcome"] - active["market_price"])).to_numpy()
    hac = mean_tstat_hac(pd.Series(realized)) if len(realized) else {"mean_bps": np.nan, "tstat": np.nan, "n": 0}

    oracle_sign = np.sign(df["true_prob"] - df["market_price"])
    oracle = (oracle_sign * (df["outcome"] - df["market_price"])).to_numpy()
    oracle_hac = mean_tstat_hac(pd.Series(oracle))

    return {
        "n_active": int(len(active)),
        "n_total": int(len(df)),
        "pct_active": float(len(active) / max(len(df), 1)),
        "machine_edge_pp": float(realized.mean() * 100) if len(realized) else np.nan,
        "machine_t": float(hac["tstat"]),
        "oracle_edge_pp": float(oracle.mean() * 100),
        "oracle_t": float(oracle_hac["tstat"]),
    }


# --------------------------------------------------------------------------- #
# 2. Noise, not signal: the raw MC edge shrinks as history grows
# --------------------------------------------------------------------------- #

def edge_vs_history(hist_lens=(20, 40, 60, 120, 250), n_markets: int = 800,
                    days_to_expiry: int = 30, seed: int = 0,
                    system: MarkovMintSystem | None = None) -> pd.DataFrame:
    """Std and mean of the *raw* Monte-Carlo edge as a function of price-history length.

    For each history length we regenerate a fresh basket of efficient markets, run the machine,
    and summarise the raw edge ``raw_prob - price``. Real information would not vanish with more
    data; estimation noise from a sparse transition matrix does. We expect ``std`` to fall (more
    transitions per cell) and ``mean`` to sit near zero throughout — i.e. the chain's only output
    is noise.
    """
    system = system or MarkovMintSystem()
    rows = {}
    for i, L in enumerate(hist_lens):
        mk = data.efficient_markets(n_markets=n_markets, hist_len=L,
                                     days_to_expiry=days_to_expiry, seed=seed + i)
        df = analyze_markets(mk, system, seed=seed + i)
        e = df["raw_edge"].to_numpy()
        rows[int(L)] = {
            "mean_raw_edge_pp": float(e.mean() * 100),
            "std_raw_edge_pp": float(e.std() * 100),
            "frac_positive": float((e > 0).mean()),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "hist_len"
    return out


# --------------------------------------------------------------------------- #
# 3. Inertness: delete the chain, nothing changes
# --------------------------------------------------------------------------- #

def inertness(markets, system: MarkovMintSystem | None = None, seed: int = 0) -> dict:
    """Compare the full pipeline vs the Markov-ablated one (``raw_prob := price``).

    If the Markov + Monte-Carlo stage carried information, deleting it would change the trade
    decisions and the system edge. We report: the fraction of markets where the *direction*
    agrees between full and ablated runs, the correlation of the two system edges, and how
    often each version takes a position. Note the ablated system edge *is*, identically, the
    pure calibration wedge ``calibrate(price) - price`` (set ``raw_prob = price`` in the
    pipeline and nothing else is left) — so a single correlation answers both "is the chain
    just the price?" and "is the edge just the calibration curve?"; there is no second,
    independent check to run. Numbers near 1 mean the chain is decorative.
    """
    system = system or MarkovMintSystem()
    full = analyze_markets(markets, system, ablate_markov=False, seed=seed)
    abla = analyze_markets(markets, system, ablate_markov=True, seed=seed)

    same_dir = float((full["direction"] == abla["direction"]).mean())
    # By construction abla["system_edge"] == calibrate(price) - price, exactly.
    edge_corr = float(np.corrcoef(full["system_edge"], abla["system_edge"])[0, 1])

    # How often each version even takes a position, and how often they pick the same side.
    return {
        "n": int(len(full)),
        "same_direction_frac": same_dir,
        "edge_corr_full_vs_ablated": edge_corr,
        "active_frac_full": float((full["dir_sign"] != 0).mean()),
        "active_frac_ablated": float((abla["dir_sign"] != 0).mean()),
    }


# --------------------------------------------------------------------------- #
# 4. Tradability: Kelly-size the bets, score vs truth, net of the spread
# --------------------------------------------------------------------------- #

def _trade_return(dir_sign: float, yes_price: float, outcome: int, spread: float) -> float:
    """Return on capital for one binary-contract trade held to resolution.

    ``spread`` is the full quoted bid/ask width; the taker crosses it **once**, on entry, so
    the toll charged is the half-spread ``spread/2`` (resolution pays face value — there is
    no exit trade and no exit spread). BUY YES lifts the ask ``yes_price + spread/2`` and is
    paid 1 iff the event resolves YES; BUY NO lifts the NO ask ``(1 - yes_price) + spread/2``
    and is paid 1 iff it resolves NO. Return is ``(payoff - effective_cost) / effective_cost``
    — i.e. on the capital staked. The cost is deliberately *not* capped below 1: near the top
    tick the ask can exceed par, and such a trade is simply a guaranteed loss (an earlier cap
    at 0.99 let the taker buy *below* the quoted price up there, manufacturing fake edge).
    """
    if dir_sign == 0.0:
        return 0.0
    if dir_sign > 0:
        eff = yes_price + spread / 2.0
        payoff = float(outcome)
    else:
        eff = (1.0 - yes_price) + spread / 2.0
        payoff = float(1 - outcome)
    return (payoff - eff) / eff


def pnl_sim(df: pd.DataFrame, spread: float = 0.02, seed: int = 0) -> dict:
    """Score the machine's bets against the true outcome, net of the bid/ask toll.

    ``spread`` is the full quoted bid/ask width (price units); each trade pays the **half-
    spread** on entry and nothing on resolution (see :func:`_trade_return`). Per-trade
    returns on capital across all *active* trades; the win rate; a per-trade Sharpe
    with a bootstrap CI (``periods_per_year=1`` — these are trades, not days); and the terminal
    bankroll from compounding the machine's own quarter-Kelly fractions. A genuine edge grows
    the bankroll; betting noise (efficient null) bleeds it, faster once the spread bites.
    """
    active = df[df["dir_sign"] != 0.0]
    rets = np.array([
        _trade_return(s, c, o, spread)
        for s, c, o in zip(active["dir_sign"], active["market_price"], active["outcome"])
    ])
    # Quarter-Kelly compounding, in the markets' original order.
    bankroll = 1.0
    for f, r in zip(active["kelly_frac"], rets):
        bankroll *= 1.0 + f * r

    if len(rets):
        sh = sharpe_ci_bootstrap(pd.Series(rets), periods_per_year=1, seed=seed)
    else:
        sh = {"sharpe": np.nan, "ci_low": np.nan, "ci_high": np.nan, "frac_negative": np.nan}

    return {
        "n_trades": int(len(rets)),
        "mean_ret": float(rets.mean()) if len(rets) else np.nan,
        "median_ret": float(np.median(rets)) if len(rets) else np.nan,
        "win_rate": float((rets > 0).mean()) if len(rets) else np.nan,
        "per_trade_sharpe": float(sh["sharpe"]),
        "sharpe_ci": (float(sh["ci_low"]), float(sh["ci_high"])),
        "sharpe_frac_negative": float(sh["frac_negative"]),
        "terminal_bankroll": float(bankroll),
    }


def pnl_by_price_bucket(df: pd.DataFrame, spread: float = 0.0,
                        bins=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)) -> pd.DataFrame:
    """Per-trade return by price bucket — locates *where* the machine's risk concentrates.

    The article's calibration table tops out at 0.958, so a contract trading richer than that
    is mechanically handed a calibrated probability *below* its own price, forcing a BUY NO.
    On a fair market that is a short of a strong favorite: zero expected value (the price is
    honest) but lottery-ticket variance — a ~2¢ stake that is lost ~98% of the time and pays
    ~49× otherwise. This breakdown makes the concentration visible (the top bucket carries the
    most trades and the most extreme return profile) and reproducible.
    """
    active = df[df["dir_sign"] != 0.0].copy()
    active["ret"] = [
        _trade_return(s, c, o, spread)
        for s, c, o in zip(active["dir_sign"], active["market_price"], active["outcome"])
    ]
    active["bucket"] = pd.cut(active["market_price"], list(bins))
    g = active.groupby("bucket", observed=False).agg(
        n=("ret", "size"),
        mean_ret=("ret", "mean"),
        win_rate=("ret", lambda x: float((x > 0).mean())),
        frac_buy_no=("direction", lambda d: float((d == "BUY NO").mean())),
        mean_cal_prob=("cal_prob", "mean"),
    )
    return g


def calibration_ceiling_effect(df: pd.DataFrame) -> dict:
    """Quantify the forced-NO bug: how many markets price above the table's ceiling, and how
    often the machine therefore shorts the favorite."""
    ceiling = max(CALIBRATION.values())
    floor = min(CALIBRATION.values())
    above = df[df["market_price"] > ceiling]
    below = df[df["market_price"] < floor]
    return {
        "ceiling": float(ceiling),
        "floor": float(floor),
        "n_above_ceiling": int(len(above)),
        "frac_above_ceiling": float(len(above) / max(len(df), 1)),
        "frac_above_that_buy_no": float((above["direction"] == "BUY NO").mean()) if len(above) else np.nan,
        "n_below_floor": int(len(below)),
    }


def cost_sweep(df: pd.DataFrame, spreads=(0.0, 0.01, 0.02, 0.04, 0.08), seed: int = 0) -> pd.DataFrame:
    """Mean net per-trade return, win rate and terminal bankroll across bid/ask spreads."""
    rows = {}
    for sp in spreads:
        r = pnl_sim(df, spread=sp, seed=seed)
        rows[float(sp)] = {
            "mean_ret": r["mean_ret"],
            "win_rate": r["win_rate"],
            "terminal_bankroll": r["terminal_bankroll"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "spread"
    return out


# --------------------------------------------------------------------------- #
# 5. Machinery sanity / could-you-trade-it: a *planted* edge, and where it lives
# --------------------------------------------------------------------------- #

def recover_planted(markets, system: MarkovMintSystem | None = None,
                    spread: float = 0.02, seed: int = 0) -> dict:
    """On markets with a real planted wedge, what does the machine recover — and at what cost?

    Three benchmarks scored against the true outcome:

      * the **machine** — the full pipeline, gross and net of the half-spread;
      * the **forced oracle** — bets ``sign(true_prob - price)`` on *every* market, cost-blind.
        This is the most *gross* edge any method could capture, but as a *net* benchmark it is
        a strawman: it is made to pay the toll even where the wedge is thinner than the toll;
      * the **cost-aware oracle** — the honest perfect-information reference: it trades a
        market only when the expected return on its side is still positive *after* the entry
        half-spread, and passes otherwise. The gap between its selectivity (how few markets
        clear the toll) and its net return is the real ceiling on tradability.

    Units: ``*_edge_pp_gross`` are probability **percentage points** captured per trade
    (``E[sign · (outcome − price)] × 100``); ``*_mean_ret_net`` are fractional **returns on
    capital staked**, net of the entry half-spread. We also repeat the machine-vs-ablated
    direction agreement from :func:`inertness`, to ask whether the chain (rather than the
    calibration table) contributes anything once a real edge is present.
    """
    system = system or MarkovMintSystem()
    df = analyze_markets(markets, system, seed=seed)
    abla = analyze_markets(markets, system, ablate_markov=True, seed=seed)

    # Machine (full and Markov-ablated), gross and net of the half-spread.
    head = headline_test(df)
    abla_head = headline_test(abla)
    pnl = pnl_sim(df, spread=spread, seed=seed)

    px = df["market_price"].to_numpy()
    tp = df["true_prob"].to_numpy()
    oc = df["outcome"].to_numpy()

    # Forced oracle: the true side on every market, cost-blind.
    osign = np.sign(tp - px)
    oracle_gross = float((osign * (oc - px)).mean() * 100)
    oracle_net = np.array([
        _trade_return(s, c, o, spread) for s, c, o in zip(osign, px, oc) if s != 0
    ])

    # Cost-aware oracle: trade the true side only if it still pays after the entry toll.
    exp_yes = (tp - (px + spread / 2.0)) / (px + spread / 2.0)
    exp_no = ((1.0 - tp) - ((1.0 - px) + spread / 2.0)) / ((1.0 - px) + spread / 2.0)
    exp_net = np.where(osign > 0, exp_yes, exp_no)
    aware = (osign != 0) & (exp_net > 0)
    aware_gross = float((osign[aware] * (oc[aware] - px[aware])).mean() * 100) if aware.any() else np.nan
    aware_net = np.array([
        _trade_return(s, c, o, spread) for s, c, o in zip(osign[aware], px[aware], oc[aware])
    ])
    aware_hac = (mean_tstat_hac(pd.Series(aware_net)) if len(aware_net)
                 else {"tstat": np.nan})

    same_dir = float((df["direction"] == abla["direction"]).mean())

    # Where the edge lives: forced-oracle gross edge by price bucket.
    bucket = pd.cut(df["market_price"], bins=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    by_bucket = pd.Series(osign * (oc - px)).groupby(bucket.reset_index(drop=True), observed=False).mean() * 100

    return {
        "machine_edge_pp_gross": head["machine_edge_pp"],
        "machine_t": head["machine_t"],
        "machine_mean_ret_net": pnl["mean_ret"],
        "machine_terminal_bankroll_net": pnl["terminal_bankroll"],
        "machine_n_trades": pnl["n_trades"],
        "ablated_edge_pp_gross": abla_head["machine_edge_pp"],
        "ablated_t": abla_head["machine_t"],
        "ablated_n_trades": abla_head["n_active"],
        "oracle_forced_edge_pp_gross": oracle_gross,
        "oracle_forced_mean_ret_net": float(oracle_net.mean()) if len(oracle_net) else np.nan,
        "oracle_aware_n_trades": int(aware.sum()),
        "oracle_aware_frac": float(aware.mean()),
        "oracle_aware_edge_pp_gross": aware_gross,
        "oracle_aware_mean_ret_net": float(aware_net.mean()) if len(aware_net) else np.nan,
        "oracle_aware_t": float(aware_hac["tstat"]),
        "same_direction_machine_vs_ablated": same_dir,
        "oracle_edge_by_price_bucket_pp": {str(k): round(float(v), 3) for k, v in by_bucket.items()},
    }
