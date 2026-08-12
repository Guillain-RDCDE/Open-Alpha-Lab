"""Study 861 — Debt-Maturity Rollover Risk.

Do firms funded with a high share of SHORT-TERM debt — a big current-maturities /
short-borrowings wall relative to their total borrowings — subsequently UNDER-earn, as the
rollover-risk story says (they must refinance into whatever rates and credit conditions prevail,
and that bites hardest when rates rise / credit tightens)?

Signal (point-in-time, from EDGAR): short-term-debt share
``st_share = (DebtCurrent + LongTermDebtCurrent) / (DebtCurrent + LongTermDebtCurrent +
LongTermDebtNoncurrent)`` — the fraction of a firm's debt maturing within a year.
"""
