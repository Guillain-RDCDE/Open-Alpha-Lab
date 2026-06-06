"""Offline demo — reproduces the whole argument with NO network.

Run:  python examples/run_synthetic_demo.py

It prints three blocks, exactly the LinkedIn narrative:
  (A) Compounding  — why a tiny innocent bias becomes "billions of percent"
  (B) Artefact     — how a few bad closes manufacture a fake overnight signal
  (C) Costs        — the gross edge (nice Sharpe) goes NEGATIVE after real fees
and saves a Figure-1(c)-style chart of the synthetic market.
"""

from __future__ import annotations

import os
import sys

# Windows consoles default to cp1252 and mangle the em-dashes below.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: never pop a window from a script

from quantlab import backtest, decompose, diagnostics, plots  # noqa: E402

OUT_PNG = "out_synthetic_decomposition.png"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    # ------------------------------------------------------------------ (A)
    section("(A) COMPOUNDING — a constant overnight bias, by horizon")
    table = diagnostics.compounding_table()
    print(diagnostics.format_compounding(table).to_string())
    print(
        "\n  -> The explosion is the EXPONENT, not fraud. A 1 bps/night drift\n"
        "     (utterly innocent) already compounds to triple digits over 30y."
    )

    # The honest 'clone' of Knuteson's chart: tiny bias + pure noise, no fraud.
    ohlc = diagnostics.synthetic_ohlc(
        overnight_bias_bps=3.0, intraday_bias_bps=-1.0, seed=0
    )
    dec = decompose.decompose(ohlc)
    s = decompose.summary(dec)
    print("\n  Synthetic market (night +3 bps, day -1 bp, pure noise):")
    print(
        f"     overnight cumulative {s.loc['overnight','cum_return']*100:+,.0f}%   "
        f"intraday {s.loc['intraday','cum_return']*100:+,.0f}%   "
        f"buy&hold {s.loc['close_close','cum_return']*100:+,.0f}%"
    )
    print(f"     identity max error = {decompose.check_identity(dec):.2e} (should be ~0)")

    ax = plots.plot_decomposition(dec, title="Synthetic market — no fraud, just a 3 bps night bias")
    ax.figure.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"     figure saved -> {OUT_PNG}")

    # ------------------------------------------------------------------ (B)
    section("(B) DATA ARTEFACT — a few bad closes fabricate an overnight signal")
    flat = diagnostics.synthetic_ohlc(
        overnight_bias_bps=0.0, intraday_bias_bps=0.0, daily_vol_bps=80.0, seed=1
    )
    dec_clean = decompose.decompose(flat)
    corrupt = diagnostics.inject_split_artifact(flat, factor=1.5)
    dec_dirty = decompose.decompose(corrupt)
    on_clean = dec_clean["cum_overnight"].iloc[-1] * 100
    on_dirty = dec_dirty["cum_overnight"].iloc[-1] * 100
    print(f"  overnight cumulative  clean: {on_clean:+,.1f}%   ->  corrupted: {on_dirty:+,.1f}%")
    flags = diagnostics.flag_suspicious_returns(dec_dirty, threshold=0.40)
    print(f"  artefact detector flagged {len(flags)} suspicious day(s):")
    print(flags.head().to_string())
    print(
        "\n  -> Mis-adjusted prices move return FROM the day leg INTO the night leg.\n"
        "     This is the mechanism behind Knuteson's wildest emerging-market plots."
    )

    # ------------------------------------------------------------------ (C)
    section("(C) COSTS — the gross edge does not survive real fees")
    be = backtest.breakeven_cost_bps(dec)
    print(f"  break-even round-trip cost = {be:.2f} bps/night")
    sweep = backtest.cost_sweep(dec)
    pretty = sweep.copy()
    pretty["cagr_net"] = (pretty["cagr_net"] * 100).map("{:+.2f}%".format)
    pretty["sharpe_net"] = pretty["sharpe_net"].map("{:+.2f}".format)
    pretty["max_drawdown"] = (pretty["max_drawdown"] * 100).map("{:.1f}%".format)
    print(pretty.to_string())
    print(
        "\n  -> A realistic ~5 bps round-trip turns a positive gross Sharpe into a\n"
        "     net LOSER — exactly what sank the NSPY / NIWM 'night effect' ETFs."
    )

    print("\nDone. This entire run used no network. See README for the live path.\n")


if __name__ == "__main__":
    main()
