"""Study 854 — Cash Conversion Cycle 🔄

The working-capital-efficiency claim: a firm's **Cash Conversion Cycle** —

    CCC = DSO + DIO − DPO

(days sales outstanding + days inventory outstanding − days payables outstanding) — is
how long cash is tied up in operations between paying suppliers and collecting from
customers. A firm that **shortens** its CCC frees cash it can redeploy and may out-earn;
a bloated or **rising** CCC is a working-capital drag. We rank filers on the year-over-year
change in CCC and go **long the shortening (falling-CCC) / short the bloating (rising-CCC)**
names.

Components, point-in-time on the 10-Q/10-K filing date (no look-ahead):
  DSO = ``AccountsReceivableNetCurrent`` ÷ (annualised ``Revenues`` ÷ 365)
  DIO = ``InventoryNet``                 ÷ (annualised ``CostOfRevenue`` ÷ 365)
  DPO = ``AccountsPayableCurrent``       ÷ (annualised ``CostOfRevenue`` ÷ 365)
with ``CostOfGoodsAndServicesSold`` as the COGS fallback.
"""
