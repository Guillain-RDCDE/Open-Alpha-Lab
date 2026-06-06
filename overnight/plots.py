"""Plotting — Knuteson's Figure 1(c) style, but with numbers in plain sight.

Knuteson plots cumulative overnight vs intraday on a *log* axis, which is where
"billions of percent" lives. We reproduce that format (it is the honest way to
show 30 years of compounding) but always print the final magnitudes in clear
text and offer a ``linear`` toggle so a beginner isn't misled by the log scale.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; callers can override before importing
import matplotlib.pyplot as plt  # noqa: E402


def plot_decomposition(dec, title: str = "Overnight vs Intraday", linear: bool = False, ax=None):
    """Plot cumulative overnight / intraday / close-close from a decomposition.

    ``linear=False`` (default) uses a log-scaled growth-of-1 axis like Figure
    1(c); ``linear=True`` plots cumulative percent return on a linear axis.
    Returns the matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if linear:
        ax.plot(dec.index, dec["cum_overnight"] * 100, label="Overnight (close→open)", lw=1.8)
        ax.plot(dec.index, dec["cum_intraday"] * 100, label="Intraday (open→close)", lw=1.8)
        ax.plot(dec.index, dec["cum_close_close"] * 100, label="Buy & hold (close→close)",
                lw=1.2, color="grey", alpha=0.8)
        ax.set_ylabel("Cumulative return (%)")
    else:
        ax.plot(dec.index, 1 + dec["cum_overnight"], label="Overnight (close→open)", lw=1.8)
        ax.plot(dec.index, 1 + dec["cum_intraday"], label="Intraday (open→close)", lw=1.8)
        ax.plot(dec.index, 1 + dec["cum_close_close"], label="Buy & hold (close→close)",
                lw=1.2, color="grey", alpha=0.8)
        ax.set_yscale("log")
        ax.set_ylabel("Growth of $1 (log scale)")

    on = dec["cum_overnight"].iloc[-1] * 100
    idd = dec["cum_intraday"].iloc[-1] * 100
    cc = dec["cum_close_close"].iloc[-1] * 100
    ax.set_title(
        f"{title}\nFinal: overnight {on:+,.0f}%   intraday {idd:+,.0f}%   "
        f"buy&hold {cc:+,.0f}%",
        fontsize=11,
    )
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Date")
    return ax


def plot_grid(decs: dict, linear: bool = False, ncols: int = 2):
    """Grid of decompositions (one panel per ticker), Figure-3 style.

    ``decs`` maps a label -> decomposition DataFrame. Returns the Figure.
    """
    n = len(decs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    for ax, (label, dec) in zip(axes.flat, decs.items()):
        plot_decomposition(dec, title=label, linear=linear, ax=ax)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig
