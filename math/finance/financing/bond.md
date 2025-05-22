# Bond

## Bond Issuance

When a company needs financing (500 mil), it issues bond by credit.
Then financial institutions bid to help bond issuance with proposed rates.

|Institutions|Proposed Rates|Quantity (million)|
|-|-|-|
|Institution A|3.975%|150|
|Institution B|4.00%|100|
|Institution C|3.95%|250|

Financial institutions will pay cash to the company, then distribute the bond per each 100 face value to individual clients.

## Bond Related Trading

* Bond trading
* Repurchase Agreement (REPO) (by pledge)
* Repurchase Agreement (REPO) (outright)
* Bond buy/sell back

## Bond Types

### Classification by Interest Payment

* Fixed-Rate Bonds

Interest Payment: Pay a fixed interest rate (coupon) periodically (usually semi-annually or annually).

* Floating-Rate Bonds (FRBs)

Interest Payment: Pay variable interest linked to a benchmark rate (e.g., LIBOR, SOFR, EURIBOR) plus a fixed spread.

* Zero coupon bonds

Interest Payment: No periodic interest. Issued at a discount and redeemed at face value.

* Step-up/down bonds

Step-Up Bonds: Interest rate increases at predefined intervals.

Step-Down Bonds: Interest rate decreases over time.

* Deferred interest bonds

Interest Payment: No interest paid for an initial period; then periodic payments begin.

* Payment-in-Kind (PIK) Bonds

Interest Payment: Pay interest in additional bonds or equity instead of cash.

### Classification by Credibility

* Rate/Govt Bond

Govt bonds are usually high quality, seldomly default

* Credit/Corp Bond

Corp bonds have higher risks of default

## Bond Basic Stats

* Coupon Factor

The Factor to be used when determining the amount of interest paid by the issuer on coupon payment dates.

* Coupon Rate

The interest rate on the security or loan-type agreement, e.g., $5.25\%$. In the formulas this would be expressed as $0.0525$.

* Day Count Factor

Figure representing the amount of the Coupon Rate to apply in calculating Interest.

* Pool factor

A pool factor is the outstanding principle out of the amount of the initial principal for ABS or MBS.

$$
F_{pool} = \frac{\text{OutstandingPrincipleBalance}}{\text{OriginalPrincipleBalance}}
$$

E.g., $F_{pool}=0.4$ for $ \$ 1,000,000 $ loan means the underlying mortgage loan that remains in a mortgage-backed security transaction is $ \$ 400,000$, and $ \$ 600,000 $ has been repaid.

## Day Count Factor: Day count Conventions

A day count convention determines how interest accrues over time.

In U.S., there are

* Actual/Actual (in period): T-bonds

$$
DayCountFactor=
\left\{
    \begin{array}{cc}
        \frac{AccrualDays}{365} &\quad \text{non-leap years}
        \\
        \frac{AccrualDays}{366} &\quad \text{leap years}
    \end{array}
\right.
$$

* 30/360: U.S. corporate and municipal bonds

$$
DayCountFactor=
\frac{
    360 \times AccrualYears
    + 30 \times AccrualMonthsOfThisYear
    + ArrualDaysOfThisMonth
}{360}
$$

* Actual/360: T-bills and other money market instruments (commonly less than 1-year maturity)

$$
DayCountFactor=
\frac{AccrualDays}{360}
$$

## Terminologies in Bonds

### Bond Issue Price/Size/Factor

* Issue Price

Usually, issue price for a bond is $100$ same as bond face value.
However, some zero-coupon bonds have $\text{IssuePrice}<100$.

It can be used for profit and tax calculation.

$$
\begin{align*}
\text{profit}&=100-\text{issuePrice} \\
\text{withholdingTax}&=\text{profit} \times \text{taxRate}
\end{align*}
$$

* Issue Size

Total bond issuance size.

An individual trader's position of a bond indicates how much exposure/manipulation he/she is to the market.
For example, if a trader has a high position of a bond $\frac{\text{traderThisBondPosition}}{\text{thisBondTotalIssueSize}}>90\%$, he/she could very much manipulate the price of this bond.

* Issue Factor

A custom discount factor to issue price, usually 100.

### Bond Pool Factor

Pool factor is used for amortizing lent securities.

Below code simulates for mortgage-based securities assumed the underlying is fully paid off after 12 months, how pool factor is updated every month.

```py
remaining_principal = original_principal
monthly_rate = annual_interest_rate / 12
monthly_payment = original_principal * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
for month in range(0, 12):
    remaining_principal -= monthly_payment
    pool_factor = remaining_principal / original_principal
```

### Bond Value/Price Calculation

* Accrued interest

Accrued interest is the interest on a bond or loan that has accumulated since the principal investment, or since the previous coupon payment if there has been one already.

In other words, the interest accounts for the time since the bond's start date or the last coupon payment date.

* Clean and dirty price

"Clean price" is the price excluding any interest that has accrued.

"Dirty price" (or "full price" or "all in price" or "Cash price") includes accrued interest.

$$
\text{dirtyPrice}=\text{poolFactor}\times(\text{cleanPrice}+\text{accruedInterest})
$$

* Value and value factor

A value factor is a custom bond value adjustment factor, usually 100.

$$
\text{bondPositionValue}=
\text{dirtyPrice} \times \text{Quantity} \div \text{issueFactor} \times \text{valueFactor}
$$

* Cash Flow and Present Value

Let $C_{t_1},C_{t_2},...,C_{t_n}$ be bond cash flow, the present value estimate is

$$
PV=\sum^n_{i=1} C_{t_i} e^{-r(t_i)t_i}
$$

For example, a three-year maturity bond with 3% annualized coupon rate would see cash flow:

$$
C_{t_1}=3,\quad C_{t_2}=3,\quad C_{t_3}=103
$$
