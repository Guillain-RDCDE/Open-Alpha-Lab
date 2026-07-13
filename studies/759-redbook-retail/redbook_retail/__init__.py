"""Study 759 — Redbook-Retail (does accelerating same-store sales lead retail stocks?).

The consumer-nowcasting folklore: the weekly Johnson Redbook Index of same-store retail
sales is a real-time read on the shopper, so when its year-over-year growth **accelerates**,
the retail sector (XRT) is supposedly about to follow — a nowcast you can trade. We rebuild
the believers' momentum signal on the monthly Redbook YoY tape (a hardcoded, clearly-labelled
approximate reconstruction, since the weekly Redbook series is proprietary and off FRED) and
measure forward 1/3/6/12-month XRT returns conditional on accelerating vs decelerating
same-store growth, against the unconditional base rate, with a Welch t, a placebo null, a
lead/lag scan, a level-regime split, a retail-vs-market relative test, and a tradable timing
overlay.

The decisive finding is about *timing, contamination and tradability*: Redbook is a nominal,
inflation-tangled same-store number that co-moves with retail at low frequency but does not
lead XRT cleanly — a market that reprices retail in real time already knows what a lagged
monthly sales gauge is about to say.

See :mod:`redbook_retail.data` (hardcoded Redbook proxy + XRT/SPY loader + deterministic
synthetic control) and :mod:`redbook_retail.strategy` (momentum signal, forward-return
inference, placebo null, lead/lag, regime split, relative test, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
