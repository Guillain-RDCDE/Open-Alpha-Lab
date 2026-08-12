"""Study 853 — Days-Sales-Outstanding Buildup 🧾

The Abarbanell-Bushee fundamental red flag: when a firm's **receivables grow faster than
its sales** — i.e. **Days Sales Outstanding (DSO) is rising** — it is a warning sign
(channel-stuffing, aggressive revenue recognition, weakening collections) that tends to
precede *lower* future earnings and returns. We rank filers on the year-over-year change in
DSO and go **long the low-change / short the high-change** names.

DSO = ``AccountsReceivableNetCurrent`` ÷ (annualised ``Revenues`` ÷ 365), point-in-time on
the 10-Q/10-K filing date (no look-ahead).
"""
