"""Study 28 — Carousel: does chasing the hot sectors beat just holding all of them?

The eleventh study mined from Kakushadze & Serur's *151 Trading Strategies* (strategy 4.1, sector
momentum rotation). The steelman: rank the equity sectors by trailing momentum and rotate into the
leaders, on the premise that a hot sector stays hot (Moskowitz-Grinblatt 1999). We run it through the
desk's protocol and ask the only question that matters for a *concentrated* bet: does it beat simply
holding all eleven sectors equal-weight? The reusable pieces:

    * :mod:`data` — a synthetic sector panel with a *persistent* per-sector relative drift (so leaders
      persist) and a no-momentum null, plus a cache-only reader for the 11 SPDR sector ETFs.
    * :mod:`rotation` — the engine: the trailing-momentum score and :func:`rotation.rotation_strength`
      (do the top-momentum sectors out-earn the bottom?).
    * :mod:`strategy` — the long-only top-``k`` rotation book and the long-short factor vs the
      equal-weight basket, monthly, net of cost.
    * :mod:`decompose` — the inference: the rotation **alpha over equal-weight** (HAC), the long-short
      factor's t-stat, and sub-sample decay.
    * :mod:`extension` — the beat-7 worked complement: a **top-k sweep** -- does rotation beat the basket
      across the number of sectors held, or only at one cherry-picked ``k``?
"""
