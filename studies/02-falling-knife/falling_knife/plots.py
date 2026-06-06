"""Plotting helpers — figures for both audiences.

Every function returns the Matplotlib ``Figure`` and (when given ``path``) saves
a PNG. We use the non-interactive Agg backend so the examples render head-less.
Numbers are always printed in the clear on the chart: a "buy the dip" curve is
only honest if the reader can see the bounce *and* the dispersion behind it.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # head-less; safe for scripts and CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_event_study(result: dict, title: str = "Event study around -3% day", path: str | None = None):
    """Average forward path with a ±1 s.e. band and the t=0 entry marker."""
    summ = result["summary"]
    matrix = result["matrix"]
    x = summ.index.to_numpy()
    mean = summ["mean"].to_numpy()
    se = (summ["std"] / np.sqrt(max(result["n_events"], 1))).to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(0.0, color="crimson", lw=1.0, ls="--", label="event (t=0)")
    ax.plot(x, mean * 100, color="navy", lw=2, label="mean path")
    ax.fill_between(x, (mean - se) * 100, (mean + se) * 100, color="navy", alpha=0.15,
                    label="±1 s.e.")
    ax.plot(x, matrix.median().to_numpy() * 100, color="seagreen", lw=1.2, ls=":",
            label="median path")

    for h in (5, 10, 20):
        if h in summ.index:
            v = summ.loc[h, "mean"] * 100
            ax.annotate(f"+{h}d: {v:+.2f}%", (h, v), textcoords="offset points",
                        xytext=(4, 6), fontsize=8, color="navy")

    ax.set_xlabel("trading days relative to event")
    ax.set_ylabel("cumulative return since entry (%)")
    ax.set_title(f"{title}  (n={result['n_events']} events)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_equity(result, title: str = "Falling-Knife equity", path: str | None = None):
    """Equity curve (log) on top, drawdown underneath."""
    eq = result.equity
    dd = eq / eq.cummax() - 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(eq.index, eq.values, color="navy", lw=1.3)
    ax1.set_yscale("log")
    ax1.set_ylabel("equity (x, log)")
    s = result.stats
    ax1.set_title(f"{title} — CAGR {s['cagr']:.2%}, Sharpe {s['sharpe']:.2f}, "
                  f"maxDD {s['max_drawdown']:.1%}, {s['n_trades']} trades")
    ax2.fill_between(dd.index, dd.values * 100, 0, color="crimson", alpha=0.4)
    ax2.set_ylabel("drawdown (%)")
    ax2.set_xlabel("date")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_cost_sweep(sweep: pd.DataFrame, title: str = "Net edge vs entry panic-slippage",
                    path: str | None = None):
    """CAGR and Sharpe as the entry (panic) slippage rises — the killer test."""
    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax1.axhline(0.0, color="0.6", lw=0.8)
    ax1.plot(sweep.index, sweep["cagr"] * 100, "o-", color="navy", label="CAGR (%)")
    ax1.set_xlabel("entry panic-slippage (bps)")
    ax1.set_ylabel("net CAGR (%)", color="navy")
    ax2 = ax1.twinx()
    ax2.plot(sweep.index, sweep["sharpe"], "s--", color="darkorange", label="Sharpe")
    ax2.set_ylabel("net Sharpe", color="darkorange")
    ax1.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_family_heatmap(scan: pd.DataFrame, metric: str = "sharpe",
                        title: str | None = None, path: str | None = None):
    """Heatmap of a metric across the (trigger x exit) family.

    Makes the data-mining surface visible: a few hot cells in a sea of mediocre
    ones is the signature of selection, not skill.
    """
    pivot = scan.pivot(index="trigger", columns="exit", values=metric)
    fig, ax = plt.subplots(figsize=(min(2 + 0.7 * pivot.shape[1], 16), 2 + 0.5 * pivot.shape[0]))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn",
                   vmin=-np.nanmax(np.abs(pivot.to_numpy())),
                   vmax=np.nanmax(np.abs(pivot.to_numpy())))
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title(title or f"Family scan — {metric}")
    fig.colorbar(im, ax=ax, shrink=0.8, label=metric)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_event_overlay(results: dict, title: str = "Mean path by drop threshold",
                       path: str | None = None):
    """Overlay the mean forward paths of several event studies on one axis.

    ``results`` maps a label (e.g. ``"-5%"``) to an ``event_study`` dict. Lets the
    reader see at a glance that deeper drops bounce harder — the monotonicity that
    distinguishes a real effect from a round-number fluke.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(0.0, color="0.4", lw=1.0, ls="--")
    cmap = plt.get_cmap("viridis")
    items = list(results.items())
    for i, (label, res) in enumerate(items):
        summ = res["summary"]
        color = cmap(i / max(len(items) - 1, 1))
        ax.plot(summ.index, summ["mean"] * 100, lw=2, color=color,
                label=f"{label}  (n={res['n_events']})")
    ax.set_xlabel("trading days relative to event")
    ax.set_ylabel("mean cumulative return since entry (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, title="drop threshold")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_threshold_heatmap(sweep: dict, title: str = "Excess vs random day — threshold sweep",
                           path: str | None = None):
    """Heatmap of conditional excess (%) over threshold x horizon.

    Cells significant at p<0.05 are starred. A real effect varies smoothly; a lone
    hot cell among flat neighbours is the look of data-mining.
    """
    exc = sweep["excess"]
    pval = sweep["p_greater"]
    arr = exc.to_numpy(dtype=float) * 100
    vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0

    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * exc.shape[1], 1.6 + 0.6 * exc.shape[0]))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(exc.shape[1]))
    ax.set_xticklabels([f"+{c}d" for c in exc.columns])
    ax.set_yticks(range(exc.shape[0]))
    ax.set_yticklabels([f"{ix}  (n={sweep['n_events'].get(ix, '?')})" for ix in exc.index])
    for i in range(exc.shape[0]):
        for j in range(exc.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                p = pval.to_numpy(dtype=float)[i, j]
                star = "*" if (p == p and p < 0.05) else ""
                ax.text(j, i, f"{v:+.2f}%{star}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("horizon")
    ax.set_ylabel("drop threshold")
    ax.set_title(title + "   (* = p<0.05)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="excess (%)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig
