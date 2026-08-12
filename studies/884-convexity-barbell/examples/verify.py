"""Reproducible headline run for Study 884 — Convexity Barbell.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached four-ETF panel under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from barbell import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")
try:  # keep unicode dashes readable on a cp1252 Windows console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

print("# Convexity Barbell — does a duration-matched SHY+TLT barbell out-earn the IEF "
      "bullet on its extra convexity?")

if not data.have_real():
    print("(cache miss — fetching the four ETF closes once)")
    data.fetch()

closes = data.load_series()
print(f"[data] {closes.shape[1]} ETFs, {len(closes)} rows  "
      f"{closes.index.min().date()} -> {closes.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint={data.fingerprint(closes)}")
print("  SURVIVORSHIP / SHORT-HISTORY: a fixed three-bond ladder (SHY/IEF/TLT) + BIL cash; "
      "all trade continuously from before 2010, but the ladder is a design choice named on "
      "the Signal axis.")

panel = data.load_panel()
ret = st.close_returns(panel)
betas = st.empirical_durations(ret, data.BOND_TICKERS, 252).mean()
print("\n[empirical duration] trailing-252d beta of each bond to the equal-weight rates factor:")
print("  " + "  ".join(f"{k} {betas[k]:+.3f}" for k in data.BOND_TICKERS)
      + "   (monotone SHY < IEF < TLT — the duration ladder)")

book = st.barbell_book(ret, data.BOND_TICKERS, data.CASH_TICKER, 252)
h = st.barbell_stats(book)
bar = book["r_barbell"].to_numpy(); bul = book["r_bullet"].to_numpy()


def ann(r):
    return (np.prod(1.0 + r) ** (252.0 / len(r)) - 1.0) * 100.0


def vol(r):
    return np.nanstd(r, ddof=1) * np.sqrt(252) * 100.0


print(f"\nBarbell = {h['w_short']:.3f}*SHY + {h['w_long']:.3f}*TLT (duration-matched to IEF), "
      f"{h['n_days']} days")
print("# THE HEADLINE — duration-matched barbell vs the IEF bullet")
print(f"  barbell : {ann(bar):+.2f}%/yr  vol {vol(bar):.2f}%  maxDD {h['mdd_barbell']*100:.1f}%")
print(f"  bullet  : {ann(bul):+.2f}%/yr  vol {vol(bul):.2f}%  maxDD {h['mdd_bullet']*100:.1f}%  "
      f"(corr barbell,bullet = {np.corrcoef(bar,bul)[0,1]:.3f})")
print(f"  spread  : {h['spread_bps']:+.3f} bps/day  NW(10) t = {h['t_nw']:+.2f}  "
      f"one-sample t = {h['t_1s']:+.2f}  Sharpe = {h['spread_sharpe']:+.3f}")
print(f"  excess-vs-excess Sharpe (minus BIL): barbell {h['sharpe_barbell_x']:+.3f} vs "
      f"bullet {h['sharpe_bullet_x']:+.3f}  -> advantage {h['sharpe_adv']:+.3f}  "
      f"(Welch t = {h['welch_t']:+.2f})")

ci = st.bootstrap_mean_ci(book["spread"].to_numpy(), n_boot=3000)
print(f"  bootstrap spread-mean CI95 = [{ci['lo_bps']:+.3f}, {ci['hi_bps']:+.3f}] bps  "
      f"(straddles zero => no edge)")

print("\n# CONVEXITY — regress the daily spread on [1, f, f^2]")
print(f"  residual duration slope (should be ~0 if duration-matched): {h['resid_dur_slope']:+.4f}")
print(f"  convexity slope on f^2 (claim: > 0, barbell more convex): {h['conv_slope']:+.4f}  "
      f"<- EMPIRICALLY WRONG-SIGNED on the real tape")
print(f"  mean-spread split: convexity part {h['spread_conv_bps']:+.4f} + carry/drift "
      f"{h['spread_carry_bps']:+.4f} bps")
smile = st.convexity_smile(book, 5)
print("  convexity smile (mean spread by |rate move| quintile, small->big):")
print("   " + "  ".join(f"[{r.bucket}] {r.mean_spread_bps:+.3f}" for r in smile.itertuples())
      + "   (no monotone rise => no convexity capture in big moves)")

print("\n# THE 2022 TELL — the biggest rate move in the sample")
cy = st.calendar_year_table(book)
for yr in (2021, 2022, 2025):
    if yr in cy.index:
        row = cy.loc[yr]
        print(f"  {yr}: barbell {row['barbell_%']:+.2f}%  bullet {row['bullet_%']:+.2f}%  "
              f"spread {row['spread_%']:+.2f}%")
print("  => in 2022's historic selloff the duration-matched barbell UNDER-performed the "
      "bullet: convexity did not pay when yields moved most.")

print("\n# ROBUSTNESS — two eras (split 2018-01-01)")
for r in st.era_table(book, "2018-01-01"):
    print(f"  {r['era']}: n={r['n']}  spread {r['spread_bps']:+.3f} bps (NW t={r['t_nw']:+.2f}, "
          f"Sharpe {r['sharpe']:+.3f})")

print("\n# PLACEBO — permute the two barbell legs in time (break the day-by-day alignment)")
pl = st.placebo_pvalue(ret, data.BOND_TICKERS, data.CASH_TICKER, n_seeds=10, n_draws_per_seed=40)
print(f"  observed {pl['obs_bps']:+.3f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
      f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']} draws -> right-tail p = {pl['p_value']:.3f}")

print("\n# THE TIMER — barbell rebalancing costed vs the buy-and-hold bullet")
for cb in (0.5, 1.0, 2.0):
    tm = st.timer_stats(book, cost_bps=cb)
    print(f"  cost={cb:>3.1f} bp: gross {tm['gross_bps']:+.3f} -> net {tm['net_bps']:+.3f} bps/day "
          f"(turnover {tm['avg_turnover']:.4f}/day, t={tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:+.3f}, "
          f"~{tm['ann_net_pct']:+.2f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
print("  (exaggerated convexity + quiet idio so the planted structure is cleanly recoverable)")
null_t = []
for s_ in range(20):
    p0 = data.synthetic_panel(edge=0.0, seed=884 + s_, n_days=1300)
    null_t.append(st.synthetic_detect(p0)["t_nw"])
null_t = np.asarray(null_t)
print(f"  null (edge=0), 20 seeds: spread NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
p1 = data.synthetic_panel(edge=0.6, seed=884, n_days=1800)
sy = st.synthetic_detect(p1)
print(f"  planted (edge=0.6 bps/day underpriced convexity, seed 884): spread NW t = "
      f"{sy['t_nw']:+.2f}, convexity slope = {sy['conv_slope']:+.3f}, residual duration slope "
      f"= {sy['resid_dur_slope']:+.4f}")
print("  => the detector recovers a planted convexity edge and stays silent on the null; "
      "the real-tape null is a genuine absence, not a broken engine.")
