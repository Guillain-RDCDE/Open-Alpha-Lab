# References & literature map — Study 63 (Free-Fall)

## The premium and the crash

- **Carr, P., & Wu, L. (2009).** *Variance Risk Premiums.* Review of Financial Studies — the volatility
  risk premium: implied vol exceeds realised, so short-vol earns a carry.
- **Bollerslev, T., Tauchen, G., & Zhou, H. (2009).** *Expected Stock Returns and Variance Risk Premia.*
  Review of Financial Studies — the premium's size and persistence.
- **"Volmageddon" (5–6 Feb 2018)** — the inverse-VIX ETPs (XIV, SVXY) lost ~80–96% in a day; XIV was
  liquidated. The defining left-tail event for naive short-vol.
- **Vendor / belief family** — short-vol / variance carry; backlog:
  [`docs/pwb_strategies_inventory.md`](../../../docs/pwb_strategies_inventory.md).

## On harvesting it survivably

- **Israelov, R. (2019).** *Pathetic Protection / Volatility selling* literature — short-vol must be
  sized and tail-hedged; the same lesson as carry generally.
- **Open-Alpha-Lab [Study 27 Steamroller](../../27-steamroller/)** — FX carry: a real premium, a
  steamroller tail; vol-targeting *fails* on jump risk. The direct analogue.

## Data

- **Yahoo! Finance** — SVXY (ProShares Short VIX Short-Term Futures) and SPY, daily total return; SVXY's
  free history begins 2018-01 (it includes Volmageddon). The offline synthetic world generates a steady
  carry plus rare catastrophic crashes (and a no-crash null) so the skew/tail is provable offline.

*The short-vol sibling of the carry-with-a-tail studies [27 Steamroller](../../27-steamroller/) (FX) and
[59 Downhill](../../59-downhill/) (duration).*
