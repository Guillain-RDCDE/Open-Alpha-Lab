"""Study 18 — Dull-Roar: do the market's calmest stocks really beat its wildest ones?

The first study drawn from Kakushadze & Serur's *151 Trading Strategies* (strategy 3.4, the
low-volatility anomaly). The steelman is one of the most-cited facts in empirical finance: sort stocks
by past volatility and the calm decile goes on to earn a *higher* risk-adjusted return than the wild
one — the security-market line is too flat (Ang-Hodrick-Xing-Zhang 2006; Baker-Bradley-Wurgler 2011;
Frazzini-Pedersen 2014). We run it through the desk's protocol and, as ever, separate "is it real?"
from "can you bank it?". The reusable pieces, in the desk's usual split:

    * :mod:`data` — the cross-section the sort runs on: a synthetic single-factor universe with a
      **baked-in flat security-market line** (low-beta names carry positive alpha, high-beta negative),
      idiosyncratic vol proportional to beta (so a total-vol sort is a beta sort, and the null is a
      flat-Sharpe control), plus a cache-only reader that reduces the current S&P 500 to a daily-returns
      panel via :mod:`quantlab.universe`. A *flat-SML* panel (``sml_slope=0``) is the null: vol carries
      no alpha, so the sort must add nothing on a beta-adjusted basis.
    * :mod:`sort` — the engine: per-stock realized vol, look-ahead-safe decile portfolios, and the
      load-bearing :func:`sort.security_market_line` — across the universe, does a lower past vol line
      up with a higher Sharpe? If the gradient weren't there the sort would be sorting noise.
    * :mod:`strategy` — the investable books (low-vol long-only, high-vol, naive dollar-neutral
      long-short) with a two-knob cost model: ordinary rebalancing slippage *and* the annual stock-loan
      fee on the short (high-vol) leg — the friction this anomaly actually lives or dies on
      (:func:`strategy.borrow_sweep`).
    * :mod:`decompose` — the inference that earns the stamps: (1) **CAPM alpha** with HAC errors;
      (2) the **beta-neutral BAB** construction (Frazzini-Pedersen) that lifts the structural short-the-
      market drag off the naive book; (3) the honest counter — a **beta-tilt test** that levers the
      low-vol book to beta 1 and asks how much "outperformance" was just *less market risk*; (4) **leg
      attribution** — does the effect live in the unshortable high-vol leg? The verdict it lands: Signal
      `REAL`, Tradability `FRAGILE`, "Free alpha?" `BETA-TILT` — a real effect whose harvestable form is
      a modest defensive tilt, not the headline spread.
    * :mod:`extension` — the beat-7 worked complement: the **no-shorting test**. Split the beta-neutral
      alpha into a long-only defensive slice (needs no borrow) and a short-the-wild slice (needs the
      borrow the real tape may not grant), and find the borrow fee that zeroes the dollar-neutral book.
"""
