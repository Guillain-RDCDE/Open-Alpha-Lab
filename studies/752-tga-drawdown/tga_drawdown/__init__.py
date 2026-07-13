"""Study 752 — TGA-Drawdown (is a falling Treasury cash balance hidden stimulus?).

The macro-liquidity thesis: the **Treasury General Account (TGA)** — the Treasury's
checking account at the Fed — is a hidden liquidity lever. When the Treasury *draws
down* its cash balance, that money flows into the banking system as reserves (a
"liquidity injection") and supposedly lifts equities over the following weeks; when
the TGA *builds*, reserves drain and equities are supposedly pressured. We rebuild
the believers' signal on a monthly TGA tape (a hardcoded, clearly-labelled monthly
**proxy** of the weekly FRED ``WTREGEN`` / Daily Treasury Statement operating cash
balance, since FRED is firewalled here) and measure forward 1/2/3/6-month SPY returns
conditional on the TGA drawing down vs building, against the unconditional base rate,
with a Welch t, a Newey-West (HAC) predictive regression, a placebo null, a lead/lag
scan, and a tradable timing overlay.

The decisive finding is about *identification and tradability*: reserves and asset
prices share a macro backdrop, but a monthly TGA drawdown is too coincident, too
tangled with debt-ceiling episodes, and too noisy to be a tradable lever you can
allocate to.

See :mod:`tga_drawdown.data` (hardcoded TGA proxy + SPY loader + deterministic
synthetic control) and :mod:`tga_drawdown.strategy` (injection signal, forward-return
inference, HAC regression, placebo null, lead/lag, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
