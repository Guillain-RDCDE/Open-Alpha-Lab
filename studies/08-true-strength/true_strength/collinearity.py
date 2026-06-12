"""The gauntlet — the tests that decide whether the TSI is its own signal or a repaint.

Four questions, in the order they kill the claim:

  1. :func:`level_collinearity` — do the *z-scored* oscillators move together? If the
     double-smoothed TSI tracks the MACD histogram and the centred RSI bar-for-bar, the
     "true strength" read is the same shape as everything else.
  2. :func:`incremental_information` — regress the TSI on the *other two*. If its R² is near
     1, the TSI carries no information the MACD and RSI don't already span. This is the
     sharpest form of "NONE": not merely correlated, but redundant.
  3. :func:`sign_agreement` — do the three *positions* (the tradable object) agree? Two
     indicators that disagree half the time are different trades; ones that agree ~90% are
     one trade.
  4. :func:`equity_collinearity` — are the three equity curves the same curve? The final,
     money proof.

Then the tradability side: :func:`reality_check_grid` prices the implicit parameter search
(White 2000, via ``quantlab.bayes.reality_check``) — is the *best* TSI of a 24-variant grid
distinguishable from luck? Run it long/flat it certifies the embedded equity beta; run it
long/short (dollar-neutral) it tests the timing itself. :func:`cost_sweep` charges the
crossover trade real costs, and :func:`orthogonalised_tsi_edge` separates the residual's
*information* (gross) from its *friction* (net).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.analytics import mean_tstat_hac, sharpe_with_se
from quantlab.bayes import reality_check
from quantlab.stats import annualized_sharpe

from . import backtest
from . import oscillators as osc


# --------------------------------------------------------------------------- #
# 1) Are the oscillators the same shape?
# --------------------------------------------------------------------------- #

def _pair_corr(a: pd.Series, b: pd.Series) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return float("nan")
    x, y = a[m].to_numpy(float), b[m].to_numpy(float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def level_collinearity(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-name Pearson correlation between the z-scored oscillators; pooled summary.

    Returns a one-row-per-pair frame with the median and inter-quartile correlation across
    names (``tsi~macd``, ``tsi~rsi``, ``macd~rsi``). High median ≈ "same shape".
    """
    pairs = {"tsi~macd": ("z_tsi", "z_macd"), "tsi~rsi": ("z_tsi", "z_rsi"),
             "macd~rsi": ("z_macd", "z_rsi")}
    acc: dict[str, list[float]] = {k: [] for k in pairs}
    for f in frames.values():
        p = osc.oscillator_panel(f["Close"])
        for name, (ca, cb) in pairs.items():
            c = _pair_corr(p[ca], p[cb])
            if np.isfinite(c):
                acc[name].append(c)
    rows = []
    for name, vals in acc.items():
        a = np.array(vals)
        rows.append({"pair": name, "n_names": a.size,
                     "median_corr": float(np.median(a)) if a.size else float("nan"),
                     "q25": float(np.percentile(a, 25)) if a.size else float("nan"),
                     "q75": float(np.percentile(a, 75)) if a.size else float("nan")})
    return pd.DataFrame(rows).set_index("pair")


def incremental_information(frames: dict[str, pd.DataFrame]) -> dict:
    """Regress z(TSI) on z(MACD) and z(RSI) per name; report the pooled R².

    A high R² means the TSI is *spanned* by the other two — the double smoothing adds no
    information they don't already carry. We report the median per-name R² and the pooled R².
    """
    r2s = []
    pooled_y, pooled_x1, pooled_x2 = [], [], []
    for f in frames.values():
        p = osc.oscillator_panel(f["Close"]).dropna()
        if len(p) < 60:
            continue
        y = p["z_tsi"].to_numpy(float)
        X = np.column_stack([np.ones(len(p)), p["z_macd"].to_numpy(float), p["z_rsi"].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot > 0:
            r2s.append(1.0 - float(np.sum(resid ** 2)) / ss_tot)
        pooled_y.append(y); pooled_x1.append(p["z_macd"].to_numpy(float)); pooled_x2.append(p["z_rsi"].to_numpy(float))
    y = np.concatenate(pooled_y); x1 = np.concatenate(pooled_x1); x2 = np.concatenate(pooled_x2)
    X = np.column_stack([np.ones(y.size), x1, x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    pooled_r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"median_name_r2": float(np.median(r2s)) if r2s else float("nan"),
            "pooled_r2": pooled_r2, "n_names": len(r2s)}


# --------------------------------------------------------------------------- #
# 2) Do the positions / equity curves agree?
# --------------------------------------------------------------------------- #

def sign_agreement(frames: dict[str, pd.DataFrame], rule: str = "zero") -> dict:
    """Fraction of bars on which the three oscillator positions agree (pooled over names)."""
    all3, pw = [], {"tsi=macd": [], "tsi=rsi": [], "macd=rsi": []}
    for f in frames.values():
        pos = backtest.positions_by_oscillator(f["Close"], rule=rule).dropna()
        if pos.empty:
            continue
        t, m, r = pos["tsi"], pos["macd"], pos["rsi"]
        all3.append(((t == m) & (m == r)).to_numpy())
        pw["tsi=macd"].append((t == m).to_numpy())
        pw["tsi=rsi"].append((t == r).to_numpy())
        pw["macd=rsi"].append((m == r).to_numpy())
    out = {"all_three": float(np.concatenate(all3).mean()) if all3 else float("nan")}
    for k, v in pw.items():
        out[k] = float(np.concatenate(v).mean()) if v else float("nan")
    return out


def equity_collinearity(frames: dict[str, pd.DataFrame], cost_bps: float = 10.0,
                        long_short: bool = True) -> dict:
    """Correlation between the three EW strategy net-return series — the money proof."""
    funcs = {
        "tsi": lambda c: backtest.tsi_position(c, long_short=long_short),
        "macd": lambda c: backtest.macd_position(c, long_short=long_short),
        "rsi": lambda c: backtest.rsi_position(c, long_short=long_short),
    }
    series = {k: backtest.equal_weight_net(frames, fn, cost_bps=cost_bps) for k, fn in funcs.items()}
    panel = pd.DataFrame(series).dropna()
    corr = panel.corr()
    return {
        "corr_tsi_macd": float(corr.loc["tsi", "macd"]),
        "corr_tsi_rsi": float(corr.loc["tsi", "rsi"]),
        "corr_macd_rsi": float(corr.loc["macd", "rsi"]),
        "sharpe_tsi": float(backtest.annualized_sharpe(panel["tsi"])),
        "sharpe_macd": float(backtest.annualized_sharpe(panel["macd"])),
        "sharpe_rsi": float(backtest.annualized_sharpe(panel["rsi"])),
    }


# --------------------------------------------------------------------------- #
# 3) Tradability — Reality Check on the grid, and the cost sweep
# --------------------------------------------------------------------------- #

def reality_check_grid(frames: dict[str, pd.DataFrame], cost_bps: float = 10.0,
                       rule: str = "signal", seed: int = 0, n_boot: int = 2000,
                       long_short: bool = False, excess_of_bh: bool = False) -> dict:
    """White (2000) Reality Check on the best TSI of the parameter grid.

    Each grid variant becomes one column of EW net daily returns; the Reality Check builds
    the null distribution of the **best** Sharpe across the 24 columns (stationary
    bootstrap, so the serial dependence survives into the null) and returns the p-value.
    A large p means the best backtest is a search artefact, not an edge.

    The long/flat default embeds the universe's long equity drift (the book is in the
    market ~half the time), so a tiny p there certifies *being long stocks*, not the
    oscillator. Two ways to take the beta out:

      * ``excess_of_bh=True`` — score every long/flat variant on its return **in excess of
        the equal-weight buy-and-hold** of the same names (the benchmark each variant is
        implicitly sold against). The honest question becomes: does the *best* variant's
        timing beat simply holding?
      * ``long_short=True`` — symmetrise every variant (+1/−1) so the unconditional drift
        cancels inside the book itself (dollar-neutral timing).
    """
    grid = osc.tsi_grid()
    cols = {}
    for gp in grid:
        fn = lambda c, gp=gp: backtest.tsi_position(c, params=gp, rule=rule, long_short=long_short)
        cols[f"tsi_{gp.r}_{gp.s}_{gp.signal}"] = backtest.equal_weight_net(frames, fn, cost_bps=cost_bps)
    panel = pd.DataFrame(cols).dropna(how="all")
    out = {"n_variants": len(grid), "long_short": bool(long_short),
           "benchmark": "buy_and_hold" if excess_of_bh else "zero"}
    if excess_of_bh:
        bh = backtest.equal_weight_net(
            frames, lambda c: pd.Series(1.0, index=c.index), cost_bps=0.0)
        out["bh_sharpe"] = float(annualized_sharpe(bh))
        panel = panel.sub(bh.reindex(panel.index), axis=0).dropna(how="all")
    rc = reality_check(panel, n_boot=n_boot, seed=seed)
    best_col = panel.columns[rc["best_series_index"]]
    out.update({
        "best_variant": best_col,
        "best_sharpe": rc["observed_max_sharpe"],
        "reality_check_pvalue": rc["reality_check_pvalue"],
    })
    return out


def cost_sweep(frames: dict[str, pd.DataFrame], rule: str = "signal",
               cost_grid_bps=(0, 5, 10, 20, 40), long_short: bool = False) -> pd.DataFrame:
    """Mean net daily return and annualised Sharpe of the default TSI strategy vs cost.

    Built on a single zero-cost EW ledger, then the turnover cost is re-charged at each level
    — so the only thing moving across rows is the cost, not the signal.
    """
    funcs = lambda c: backtest.tsi_position(c, rule=rule, long_short=long_short)
    # build per-name gross net at zero cost AND turnover, then re-cost.
    gross_cols, turn_cols = {}, {}
    for tk, f in frames.items():
        led = backtest.run_strategy(f["Close"], funcs(f["Close"]), cost_bps=0.0)
        gross_cols[tk] = led["gross"]
        turn_cols[tk] = led["turnover"]
    gross = pd.DataFrame(gross_cols).mean(axis=1, skipna=True)
    turn = pd.DataFrame(turn_cols).mean(axis=1, skipna=True)
    rows = []
    for c in cost_grid_bps:
        net = gross - turn * (c / 1e4)
        rows.append({"cost_bps": c, "mean_net_bps": float(net.mean() * 1e4),
                     "sharpe": float(backtest.annualized_sharpe(net)),
                     "ann_turnover": float(turn.mean() * 252)})
    return pd.DataFrame(rows).set_index("cost_bps")


# --------------------------------------------------------------------------- #
# 3b) The closing argument — trade the part of the TSI MACD+RSI can't explain
# --------------------------------------------------------------------------- #

def orthogonalised_tsi_edge(frames: dict[str, pd.DataFrame], cost_bps: float = 10.0,
                            long_short: bool = True, min_rows: int = 100) -> dict:
    """Regress the TSI out of MACD+RSI per name and trade the **residual** — the part of
    the "true strength" signal that the other two cannot reproduce.

    This is the cleanest form of the redundancy verdict, and the pre-declared criterion is
    **gross**: does the residual's *information* earn anything before costs? We
    orthogonalise on the **full sample** (hindsight betas), which is the *steelman*: it
    hands the TSI the best possible incremental signal, free of any causal constraint. If
    even that residual earns a gross Sharpe of ≈ 0 — while the raw long/short TSI already
    earns ≈ 0 — the TSI's unique content carries no information MACD+RSI don't.

    The **net** numbers are reported separately and are a statement about *friction*, not
    information: the residual flips sign far more often than the level (its turnover is
    several times the raw strategy's), so any cost turns a zero-information signal into a
    reliably negative P&L. Conflating the two would let cost drag masquerade as
    "anti-signal". Gross inference comes with a Newey-West (HAC) t-stat on the mean and a
    Lo (2002) autocorrelation-adjusted SE on the Sharpe.
    """
    resid_net, resid_gross, resid_turn = {}, {}, {}
    raw_net, raw_gross, raw_turn = {}, {}, {}
    for tk, f in frames.items():
        p = osc.oscillator_panel(f["Close"]).dropna()
        if len(p) < min_rows:
            continue
        y = p["z_tsi"].to_numpy(float)
        X = np.column_stack([np.ones(len(p)), p["z_macd"].to_numpy(float), p["z_rsi"].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = pd.Series(y - X @ beta, index=p.index)
        rpos = osc.position_zero_cross(resid, long_short=long_short)
        led = backtest.run_strategy(f["Close"], rpos, cost_bps=cost_bps)
        resid_net[tk], resid_gross[tk], resid_turn[tk] = led["net"], led["gross"], led["turnover"]
        rawpos = backtest.tsi_position(f["Close"], rule="zero", long_short=long_short)
        led = backtest.run_strategy(f["Close"], rawpos, cost_bps=cost_bps)
        raw_net[tk], raw_gross[tk], raw_turn[tk] = led["net"], led["gross"], led["turnover"]
    if not resid_net:
        return {"n_names": 0}
    resid_ew = pd.DataFrame(resid_net).mean(axis=1, skipna=True)
    resid_gw = pd.DataFrame(resid_gross).mean(axis=1, skipna=True)
    raw_ew = pd.DataFrame(raw_net).mean(axis=1, skipna=True)
    raw_gw = pd.DataFrame(raw_gross).mean(axis=1, skipna=True)
    g_t = mean_tstat_hac(resid_gw)
    g_sr = sharpe_with_se(resid_gw, method="lo")
    return {
        "n_names": len(resid_net),
        # gross — the information question (the pre-declared "residual ≈ 0" criterion)
        "residual_gross_sharpe": float(annualized_sharpe(resid_gw)),
        "residual_gross_sharpe_se": float(g_sr["se_ann"]),
        "residual_gross_mean_bps": float(resid_gw.mean() * 1e4),
        "residual_gross_hac_t": float(g_t["tstat"]),
        "raw_tsi_ls_gross_sharpe": float(annualized_sharpe(raw_gw)),
        # net — the friction statement (turnover × cost, not information)
        "residual_net_sharpe": float(annualized_sharpe(resid_ew)),
        "residual_net_mean_bps": float(resid_ew.mean() * 1e4),
        "raw_tsi_ls_net_sharpe": float(annualized_sharpe(raw_ew)),
        "raw_tsi_ls_net_mean_bps": float(raw_ew.mean() * 1e4),
        "residual_ann_turnover": float(pd.DataFrame(resid_turn).mean(axis=1, skipna=True).mean() * 252),
        "raw_ann_turnover": float(pd.DataFrame(raw_turn).mean(axis=1, skipna=True).mean() * 252),
    }


# --------------------------------------------------------------------------- #
# 4) Machinery sanity on the synthetic universe
# --------------------------------------------------------------------------- #

def structure_recall(frames: dict[str, pd.DataFrame], truth, rule: str = "zero") -> dict:
    """What does the offline control actually prove? That the collinearity is **mechanical**.

    Run the full comparison machinery on names with planted momentum structure *and* on
    pure random walks, and report both sides. The honest reading has two parts:

      * **Position agreement** is a little higher on structured names than on noise —
        the trades line up best where there is a real trend to agree about.
      * **Level collinearity and the spanning R² are about as high on pure noise as on
        structure** (and as on the real data). The TSI, the MACD line and the centred RSI
        are all smoothings of the *same* recent price changes, so they co-move by
        construction — on any input whatsoever.

    The second fact is the load-bearing one: the high TSI~MACD correlation is a property
    of the **filters**, not a discovery about the market. The control therefore doesn't
    certify that the real-data collinearity is "the market talking" — it proves the
    opposite, that no market is needed to produce it. That *strengthens* the rebranding
    verdict: an indicator whose near-identity with the MACD is structural can't be a
    distinct signal on any data.
    """
    structured = {t.ticker for t in truth}
    struct_frames = {k: v for k, v in frames.items() if k in structured}
    noise_frames = {k: v for k, v in frames.items() if k not in structured}
    lc_s = level_collinearity(struct_frames)
    lc_n = level_collinearity(noise_frames)
    return {
        "agree_structured": sign_agreement(struct_frames, rule)["all_three"],
        "agree_noise": sign_agreement(noise_frames, rule)["all_three"],
        "corr_tsi_macd_structured": float(lc_s.loc["tsi~macd", "median_corr"]),
        "corr_tsi_macd_noise": float(lc_n.loc["tsi~macd", "median_corr"]),
        "spanning_r2_structured": float(incremental_information(struct_frames)["pooled_r2"]),
        "spanning_r2_noise": float(incremental_information(noise_frames)["pooled_r2"]),
    }
