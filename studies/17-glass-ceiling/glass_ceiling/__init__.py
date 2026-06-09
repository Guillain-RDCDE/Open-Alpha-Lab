"""Study 17 — Glass-Ceiling: do filtered resistance breakouts have harvestable momentum, or are you
just paying the spread twice to buy the high?

A viral retail playbook (Koroush AK's "Breakout Trading Strategy") says: long the break of resistance
after two 1-minute closes clear it, stop at the swing low (floored at 1%), take profit at **1R**, and
only when three "optimal environment" filters align — a slow staircase approach, building volume, and
a clean trend (few 30-SMMA crossovers). We mechanize it charitably and ask the desk's two questions:
is the breakout edge *real*, and would it *survive the real world*?

The decisive observation is arithmetic. A 1R-stop / 1R-target trade is a **symmetric ±1R bracket**,
whose expectancy is ``(2·win_rate − 1) − cost_R`` and whose break-even win rate is ``0.5 + cost_R/2``
— strictly above a coin flip. So the entire case rests on one measurement: is the win rate reliably
above that line? The pieces, in the desk's usual split:

    * :mod:`data` — the tape, with the answer baked in. A synthetic minute generator whose only
      load-bearing knob is post-breakout drift: ``cont_drift=0`` is the **null** (a fresh high carries
      no information, so the bracket is a coin flip *by construction*); ``cont_drift>0`` is genuine
      continuation (the steelman, where the win rate must rise); ``cont_requires_grind=True`` gates
      that continuation on a calm approach, giving the staircase filter something real to find. Plus a
      cache-only reader for real intraday bars (a small-sample sanity check — Yahoo only serves a short
      intraday window).
    * :mod:`levels` — the geometry a trader draws by eye, mechanized: trailing-high resistance, an
      N-close confirmation trigger, and the swing-low stop with Koroush's 1% floor (the floor matters —
      a too-tight stop manufactures fake stop-outs).
    * :mod:`strategy` — the bracket, resolved trade-by-trade with pessimistic intrabar fills, and the
      arithmetic that judges it: win rate with a Wilson interval, gross/net expectancy in R, the
      break-even win rate, an equity curve in R, and a cost sweep.
    * :mod:`filters` — the three environment filters as numbers (grind, volume slope, SMMA-crossover
      cleanliness) and :func:`filters.filter_lift` — does the A-grade subset beat the field, or does it
      just shrink the sample while the win rate stays put?

The verdict it lands: Signal `NONE` on the level (the breakout win rate is a coin flip — ~50% on the
null tape and on real intraday once you stop cherry-picking screenshots), Tradability `MIRAGE` (at 1:1
the 1%-stop bracket needs >50% *net*, but the spread paid twice on the minute chart pushes break-even
well past what the signal delivers), and a third axis — "Do the filters help? `NOT SUPPORTED`" — the
staircase/volume/clean-trend conditioning adds no measurable win-rate lift; it only thins the trade
count. The steelman tapes prove the test has power: when continuation is real, the same machinery
finds it.
"""
