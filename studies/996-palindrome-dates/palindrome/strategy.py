"""How easy is it to find a pattern that is not there? — Study 996.

Most studies on this desk try to avoid data mining. This one commits it deliberately, at scale,
on a hypothesis that is guaranteed false, and measures the result. The purpose is calibration:
if searching a few hundred meaningless calendar rules produces effects of a certain size, then
any *real* study that searched a similar space and found an effect of that size has found
nothing.

Palindromic dates are the ideal null hypothesis. There is no mechanism — no earnings, no
rebalancing, no option expiry, no human ritual attaches to a date reading the same backwards.
So every result below is known in advance to be spurious, and the interesting quantity is not
whether the *t*-statistic clears 2 but **how many rules had to be tried before one did**.

The module provides:

- ``date_predicates`` — roughly two hundred calendar rules of escalating silliness, from
  "palindromic in DD/MM/YYYY" through "digit sum is prime" to "day number is a Fibonacci
  number". Each is a legitimate-looking boolean function of the date and none has a mechanism.
- ``scan`` — runs every rule against every asset and returns the *t*-statistics.
- ``multiple_testing_summary`` — what the distribution of those *t*-statistics should look like
  under the null, what it actually looks like, and the Bonferroni and Benjamini-Hochberg
  thresholds that would have been needed.
- ``best_rule_distribution`` — the number that matters. Under pure noise, what is the expected
  maximum *t*-statistic from *k* tries? A researcher who reports only their best rule is
  reporting a draw from *this* distribution, not from a *t*-distribution, and the difference is
  the whole problem.
- ``tradable_check`` — takes the single best rule found and prices it honestly out of sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
def _is_palindrome(text: str) -> bool:
    return text == text[::-1]


def _digits(d: pd.Timestamp, fmt: str) -> str:
    return d.strftime(fmt)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


FIBONACCI = {1, 2, 3, 5, 8, 13, 21, 34}


def date_predicates() -> dict:
    """A catalogue of calendar rules, none of which can possibly matter.

    They are deliberately of the kind that appear in real market-lore articles: palindromes,
    repeated digits, prime days, Fibonacci days, "the digits add to seven". Each is a clean
    boolean function of the date alone, so none of them can be contaminated by the data — the
    only thing being searched is the calendar itself.
    """
    preds = {}

    preds["palindrome DDMMYYYY"] = lambda d: _is_palindrome(_digits(d, "%d%m%Y"))
    preds["palindrome MMDDYYYY"] = lambda d: _is_palindrome(_digits(d, "%m%d%Y"))
    preds["palindrome DDMMYY"] = lambda d: _is_palindrome(_digits(d, "%d%m%y"))
    preds["palindrome MMDDYY"] = lambda d: _is_palindrome(_digits(d, "%m%d%y"))
    preds["palindrome YYYYMMDD"] = lambda d: _is_palindrome(_digits(d, "%Y%m%d"))
    preds["day == month"] = lambda d: d.day == d.month
    preds["day == month == year%100"] = lambda d: d.day == d.month == (d.year % 100)
    preds["all digits same"] = lambda d: len(set(_digits(d, "%d%m%y"))) == 1
    preds["repdigit day"] = lambda d: d.day in (11, 22)

    for k in range(1, 10):
        preds[f"day is a multiple of {k}"] = (lambda k: lambda d: d.day % k == 0)(k)
    for k in range(2, 8):
        preds[f"digit sum divisible by {k}"] = (
            lambda k: lambda d: sum(int(c) for c in _digits(d, "%d%m%Y")) % k == 0)(k)
    for target in range(5, 30):
        preds[f"digit sum == {target}"] = (
            lambda t: lambda d: sum(int(c) for c in _digits(d, "%d%m%Y")) == t)(target)

    preds["prime day"] = lambda d: _is_prime(d.day)
    preds["prime month"] = lambda d: _is_prime(d.month)
    preds["prime day and month"] = lambda d: _is_prime(d.day) and _is_prime(d.month)
    preds["prime digit sum"] = lambda d: _is_prime(
        sum(int(c) for c in _digits(d, "%d%m%Y")))
    preds["Fibonacci day"] = lambda d: d.day in FIBONACCI
    preds["Fibonacci day-of-year"] = lambda d: d.dayofyear in FIBONACCI
    preds["day is a perfect square"] = lambda d: int(np.sqrt(d.day)) ** 2 == d.day
    preds["day is a power of two"] = lambda d: d.day in (1, 2, 4, 8, 16)

    for dow in range(5):
        name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][dow]
        preds[f"{name}"] = (lambda w: lambda d: d.dayofweek == w)(dow)
        for half in (True, False):
            lab = "first half" if half else "second half"
            preds[f"{name} in the {lab} of the month"] = (
                lambda w, hf: lambda d: d.dayofweek == w and ((d.day <= 15) == hf))(dow, half)
    for m in range(1, 13):
        preds[f"month {m}"] = (lambda mm: lambda d: d.month == mm)(m)
    for q in range(1, 5):
        preds[f"quarter {q}"] = (lambda qq: lambda d: (d.month - 1) // 3 + 1 == qq)(q)

    preds["even year"] = lambda d: d.year % 2 == 0
    preds["leap year"] = lambda d: (d.year % 4 == 0 and d.year % 100 != 0) or d.year % 400 == 0
    preds["year digit sum is prime"] = lambda d: _is_prime(
        sum(int(c) for c in str(d.year)))
    preds["day-of-year is prime"] = lambda d: _is_prime(d.dayofyear)
    preds["week number is odd"] = lambda d: d.isocalendar()[1] % 2 == 1
    for k in (3, 5, 7, 11, 13):
        preds[f"day-of-year divisible by {k}"] = (
            lambda kk: lambda d: d.dayofyear % kk == 0)(k)
    return preds


def apply_predicate(index: pd.DatetimeIndex, pred) -> pd.Series:
    """Evaluate one rule over a date index."""
    return pd.Series([bool(pred(d)) for d in index], index=index)


def build_mask_matrix(index: pd.DatetimeIndex, preds: dict | None = None):
    """Evaluate every rule once and return a boolean matrix (days x rules).

    The load-bearing optimisation in this study, and it is a *correctness* point as much as a
    speed one. Every predicate is a function of the **date alone**, so the mask matrix does not
    change when the returns are shuffled. Rebuilding it inside the shuffle loop would cost
    nothing but time — but it also invites the mistake of accidentally letting a rule see the
    data. Computing it once, before any returns are touched, makes that impossible.
    """
    preds = preds or date_predicates()
    names = list(preds)
    M = np.empty((len(index), len(names)), dtype=bool)
    for j, name in enumerate(names):
        f = preds[name]
        M[:, j] = [bool(f(d)) for d in index]
    return names, M


def scan_matrix(values: np.ndarray, names: list, M: np.ndarray,
                min_n: int = 20) -> np.ndarray:
    """Welch *t* for every rule at once, from a precomputed mask matrix.

    Pure linear algebra: sums and sums-of-squares under each mask give every group's mean and
    variance without a Python loop, which is what makes a few hundred shuffles of a full search
    feasible at all.
    """
    n = len(values)
    n_hit = M.sum(axis=0).astype(float)
    n_miss = n - n_hit
    total = values.sum()
    total_sq = (values ** 2).sum()
    sum_hit = values @ M
    sumsq_hit = (values ** 2) @ M
    sum_miss = total - sum_hit
    sumsq_miss = total_sq - sumsq_hit
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_hit = sum_hit / n_hit
        mean_miss = sum_miss / n_miss
        var_hit = (sumsq_hit - n_hit * mean_hit ** 2) / np.maximum(n_hit - 1, 1)
        var_miss = (sumsq_miss - n_miss * mean_miss ** 2) / np.maximum(n_miss - 1, 1)
        se = np.sqrt(var_hit / n_hit + var_miss / n_miss)
        t = (mean_hit - mean_miss) / se
    bad = (n_hit < min_n) | (n_miss < min_n) | ~np.isfinite(t)
    t[bad] = np.nan
    return t


# --------------------------------------------------------------------------- #
# Testing them
# --------------------------------------------------------------------------- #
def test_rule(rets: pd.Series, mask: pd.Series, min_n: int = 20) -> dict:
    """A Welch two-sample *t* on the days the rule selects against every other day."""
    m = mask.reindex(rets.index).fillna(False).astype(bool)
    hit, miss = rets[m], rets[~m]
    if len(hit) < min_n or len(miss) < min_n:
        return {"n_hit": int(len(hit)), "t": np.nan}
    diff = float(hit.mean() - miss.mean())
    se = float(np.sqrt(hit.var(ddof=1) / len(hit) + miss.var(ddof=1) / len(miss)))
    return {"n_hit": int(len(hit)), "n_miss": int(len(miss)),
            "mean_hit": float(hit.mean()), "mean_miss": float(miss.mean()),
            "difference": diff, "t": diff / se if se > 0 else np.nan,
            "ann_difference": diff * TRADING_DAYS}


def scan(rets: pd.Series, preds: dict | None = None, min_n: int = 20,
         cache: tuple | None = None) -> pd.DataFrame:
    """Run every rule against one return series.

    ``cache`` is an optional ``(names, mask_matrix)`` pair from ``build_mask_matrix``, reused
    across repeated scans of the same calendar.
    """
    r = rets.dropna()
    names, M = cache if cache is not None else build_mask_matrix(r.index, preds)
    if len(names) == 0 or len(r) == 0:
        return pd.DataFrame(columns=["n_hit", "n_miss", "mean_hit", "mean_miss",
                                     "difference", "t", "ann_difference"],
                            index=pd.Index([], name="rule"))
    v = r.to_numpy(dtype=float)
    t = scan_matrix(v, names, M, min_n)
    n_hit = M.sum(axis=0)
    mean_hit = np.where(n_hit > 0, (v @ M) / np.maximum(n_hit, 1), np.nan)
    mean_miss = np.where(n_hit < len(v),
                         (v.sum() - v @ M) / np.maximum(len(v) - n_hit, 1), np.nan)
    df = pd.DataFrame({"n_hit": n_hit, "n_miss": len(v) - n_hit,
                       "mean_hit": mean_hit, "mean_miss": mean_miss,
                       "difference": mean_hit - mean_miss, "t": t},
                      index=pd.Index(names, name="rule"))
    df["ann_difference"] = df["difference"] * TRADING_DAYS
    return df.dropna(subset=["t"]).sort_values("t", key=abs, ascending=False)


def scan_panel(assets: dict, preds: dict | None = None, min_n: int = 20) -> pd.DataFrame:
    """Every rule against every asset — where the multiple-testing problem really lives."""
    preds = preds or date_predicates()
    frames = []
    for name, r in assets.items():
        d = scan(r, preds, min_n)          # each asset has its own calendar, so its own cache
        if len(d):
            frames.append(d.assign(asset=name))
    return pd.concat(frames) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# The arithmetic of searching
# --------------------------------------------------------------------------- #
def multiple_testing_summary(t_values: pd.Series, alpha: float = 0.05) -> dict:
    """What the search found, against what the null says it should have found."""
    t = t_values.dropna()
    n = len(t)
    if n < 5:
        return {"n_tests": int(n)}
    p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    hits = int((p < alpha).sum())
    expected = alpha * n
    bonf = alpha / n
    order = np.sort(p)
    bh_thresh = 0.0
    for i, pv in enumerate(order, start=1):
        if pv <= alpha * i / n:
            bh_thresh = pv
    return {"n_tests": int(n), "n_significant": hits, "expected_by_luck": float(expected),
            "excess": float(hits - expected),
            "max_abs_t": float(np.abs(t).max()),
            "bonferroni_p": float(bonf),
            "bonferroni_t": float(stats.norm.ppf(1 - bonf / 2)),
            "n_surviving_bonferroni": int((p < bonf).sum()),
            "bh_threshold_p": float(bh_thresh),
            "n_surviving_bh": int((p <= bh_thresh).sum()) if bh_thresh > 0 else 0,
            "min_p": float(p.min())}


def expected_max_t(n_tests: int, n_sims: int = 4000, seed: int = 996) -> dict:
    """The distribution of the LARGEST |t| from ``n_tests`` independent null tests.

    This is the number that matters and the one nobody computes. A researcher who tries two
    hundred ideas and reports the best one is not drawing from a *t*-distribution — they are
    drawing from the distribution of the *maximum* of two hundred draws, whose median sits
    around 2.8 and whose upper tail reaches past 4. Comparing their headline *t* to 1.96 is
    comparing it to the wrong distribution entirely.
    """
    rng = np.random.default_rng(seed)
    maxes = np.abs(rng.standard_normal((n_sims, max(n_tests, 1)))).max(axis=1)
    return {"n_tests": int(n_tests), "median": float(np.median(maxes)),
            "p90": float(np.percentile(maxes, 90)),
            "p99": float(np.percentile(maxes, 99)),
            "mean": float(maxes.mean()),
            "share_above_2": float((maxes > 2).mean()),
            "share_above_3": float((maxes > 3).mean())}


def best_rule_distribution(rets: pd.Series, preds: dict | None = None,
                           n_shuffles: int = 200, seed: int = 996) -> dict:
    """The empirical version: shuffle the returns, rescan, keep the best |t|. Repeat.

    Better than the theoretical calculation because it preserves the *dependence* between rules
    — "prime day" and "day is a multiple of 3" overlap heavily, so the effective number of
    independent tests is smaller than the raw count. Shuffling handles that automatically.
    """
    preds = preds or date_predicates()
    r = rets.dropna()
    cache = build_mask_matrix(r.index, preds)
    real = scan(r, preds, cache=cache)
    if real.empty:
        return {}
    observed = float(real["t"].abs().max())
    rng = np.random.default_rng(seed)
    values = r.to_numpy(dtype=float)
    names, M = cache
    maxes = []
    for _ in range(n_shuffles):
        t = scan_matrix(rng.permutation(values), names, M)
        if np.isfinite(t).any():
            maxes.append(float(np.nanmax(np.abs(t))))
    maxes = np.array(maxes)
    return {"observed_max_t": observed, "n_shuffles": int(len(maxes)),
            "null_median_max_t": float(np.median(maxes)),
            "null_p95_max_t": float(np.percentile(maxes, 95)),
            "p_value": float((maxes >= observed).mean()),
            "best_rule": str(real.index[0]),
            "best_t": float(real.iloc[0]["t"]),
            "n_rules": int(len(real))}


def deflated_t(observed_t: float, n_tests: int) -> float:
    """The observed *t* restated against the right benchmark.

    Divides the observed *t* by the expected maximum under the null. A value above 1 means the
    finding beats what pure search would have produced; below 1 means it does not. It is a
    crude cousin of Bailey & López de Prado's deflated Sharpe ratio and it makes the point in
    one number.
    """
    e = expected_max_t(n_tests, n_sims=2000)["median"]
    return float(abs(observed_t) / e) if e > 0 else np.nan


# --------------------------------------------------------------------------- #
# What believing it costs
# --------------------------------------------------------------------------- #
def split_sample_check(rets: pd.Series, preds: dict | None = None,
                       split: float = 0.5) -> pd.DataFrame:
    """Pick the best rule in the first half; see what it does in the second.

    The oldest and still the best defence. A rule selected on one sample and tested on another
    has to survive a genuinely out-of-sample test, and calendar rules essentially never do.
    """
    preds = preds or date_predicates()
    r = rets.dropna()
    k = int(len(r) * split)
    first, second = r.iloc[:k], r.iloc[k:]
    d1 = scan(first, preds)
    if d1.empty:
        return pd.DataFrame()
    rows = []
    for rule in d1.index[:10]:
        mask2 = apply_predicate(second.index, preds[rule])
        out = test_rule(second, mask2)
        rows.append({"rule": rule, "t_in_sample": float(d1.loc[rule, "t"]),
                     "t_out_of_sample": out.get("t", np.nan),
                     "ann_in": float(d1.loc[rule, "ann_difference"]),
                     "ann_out": out.get("ann_difference", np.nan),
                     "n_out": out.get("n_hit", 0)})
    return pd.DataFrame(rows).set_index("rule")


def tradable_check(rets: pd.Series, mask: pd.Series, cash: pd.Series | None = None,
                   cost_bps: float = 2.0) -> dict:
    """Hold the market on the rule's days and cash otherwise, with costs."""
    m = mask.reindex(rets.index).fillna(False).astype(bool)
    c = (cash.reindex(rets.index).fillna(0.0) if cash is not None
         else pd.Series(0.0, index=rets.index))
    switches = m.astype(int).diff().abs().fillna(0.0)
    strat = pd.Series(np.where(m, rets, c), index=rets.index) - switches * cost_bps / 1e4
    years = len(rets) / TRADING_DAYS
    def stats_(x):
        cu = (1 + x).cumprod()
        sd = float(x.std(ddof=1))
        return {"cagr": float(cu.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
                "sharpe": float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
                "max_dd": float((cu / cu.cummax() - 1).min())}
    return {"rule_days": int(m.sum()), "share_invested": float(m.mean()),
            "switches_per_year": float(switches.sum() / years),
            "strategy": stats_(strat), "buy_hold": stats_(rets),
            "returns": strat}


def synthetic_returns(n: int = 8000, vol: float = 0.16, drift: float = 0.08,
                      seed: int = 996) -> pd.Series:
    """A pure random walk on a real trading calendar. No calendar structure at all."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1993-02-01", periods=n)
    return pd.Series(rng.normal(drift / TRADING_DAYS, vol / np.sqrt(TRADING_DAYS), n),
                     index=idx, name="ret")


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    This study is inverted: finding an effect is the *failure* mode, because the hypothesis is
    known false. So the stamps grade the **demonstration**, not the pattern.

    - **Signal**: **Busted** — the intended result — when the best rule fails the shuffle test,
      i.e. its *t* is no larger than what a search over shuffled returns produces. **Partial**
      if it beats the shuffle but fails Bonferroni. **Confirmed** would mean a palindrome effect
      genuinely survived correction, which would indicate a bug rather than a discovery, and
      the results section says so.
    - **Tradability**: **Mirage** unless a rule survives out of sample, which none should.
    """
    beats_shuffle = h["shuffle_p"] < 0.05
    survives_bonferroni = h["n_surviving_bonferroni"] > 0
    signal = ("Confirmed" if (beats_shuffle and survives_bonferroni)
              else ("Partial" if beats_shuffle else "Busted"))
    trad = ("Mirage" if h["median_oos_t"] < 2 else "Fragile")
    return {
        "signal": signal,
        "signal_why": (
            f"Searching **{h['n_rules']} meaningless calendar rules** across "
            f"{h['n_assets']} assets — {h['n_tests']} tests in total — produced "
            f"**{h['n_significant']} results significant at 5%**, against "
            f"{h['expected_by_luck']:.0f} expected by pure luck. The best was "
            f"*{h['best_rule']}* on {h['best_asset']} at *t* = **{h['best_t']:+.2f}**, worth "
            f"{h['best_ann']:+.1%} a year. On a *t*-table that is a one-in-"
            f"{h['best_naive_odds']:,.0f} event. It is nothing of the sort. Reshuffling the "
            f"returns and rerunning the identical search produced a best |*t*| of "
            f"**{h['null_median_max_t']:.2f}** at the median and {h['null_p95_max_t']:.2f} at "
            f"the 95th percentile, so the observed maximum has a shuffle *p*-value of "
            f"**{h['shuffle_p']:.2f}**. Bonferroni would have required *t* > "
            f"{h['bonferroni_t']:.2f}; **{h['n_surviving_bonferroni']}** rules cleared it. "
            f"The correct reading of the headline number is that it is a draw from the "
            f"distribution of the *maximum* of {h['n_tests']} tries, whose median is "
            f"{h['expected_max']:.2f} — not from a *t*-distribution, whose median is 0.67."),
        "trad_why": (
            f"Selecting the ten best rules on the first half of the sample and testing them on "
            f"the second half is the whole story in one table: their in-sample *t* averaged "
            f"**{h['mean_is_t']:+.2f}** and their out-of-sample *t* averaged "
            f"**{h['mean_oos_t']:+.2f}**, with {h['n_oos_survive']} of 10 keeping the same "
            f"sign. Traded with costs, the single best rule turned {h['best_ann']:+.1%} a year "
            f"of apparent edge into {h['best_traded_gap']:+.2%} against simply holding the "
            f"index. That is the price of believing a pattern with no mechanism, and it is "
            f"charged in full."),
        "trad": trad,
        "one_sentence": (
            f"{h['n_tests']} tests of hypotheses that cannot be true produced a best *t* of "
            f"{h['best_t']:+.2f} — which sounds like a discovery until you notice that "
            f"shuffling the returns produces {h['null_median_max_t']:.2f} on a median attempt."),
    }
