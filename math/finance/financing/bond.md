# Bond

## Bond Issuance

When a company needs financing (500 mil), it issues bond by credit.
Then financial institutions bid to help bond issuance with proposed rates.

|Institutions|Proposed Rates|Quantity (million)|
|-|-|-|
|Institution A|3.975%|150|
|Institution B|4.00%|100|
|Institution C|3.95%|250|

Financial institutions will pay cash to the company, then

## Bond Related Trading

* Bond trading
* Repurchase Agreement (REPO) (by pledge)
* Repurchase Agreement (REPO) (outright)
* Bond buy/sell back

## Bond Types

### Classification by Interest Payment

* Zero coupon bonds

A zero-coupon bond, also known as an accrual bond, is a debt security that does not pay interest but instead trades at a deep discount, rendering a profit at maturity, when the bond is redeemed for its full face value.

* Step-up bonds

A step-up bond is a bond that pays a lower initial interest rate but includes a feature that allows for rate increases at periodic intervals.

* Deferred interest bonds

A deferred interest bond, also called a deferred coupon bond, is a debt instrument that pays all of its interest that has accrued in the form of a single payment made at a later date rather than in periodic increments.

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

## Bond valuation (Discounted Cash Flow)

The theoretical fair value of a bond is the present value of the stream of cash flows it is expected to generate. Hence, the value of a bond is obtained by discounting the bond's expected cash flows to the present using an appropriate discount rate.

Present value $V_{present}$ can be computed by coupon payment $C$ over a number of periods $n$ with an interest rate $r$, plus the its face value $F$ (equal to its maturity value) on the final date. 

$$
V_{present} =
\big(
    \frac{C}{1+r}
    + \frac{C}{(1+r)^2}
    + \frac{C}{(1+r)^3}
    + ...
    + \frac{C}{(1+r)^n}
\big) +
\frac{F}{(1+r)^n}
$$

where coupon interest rate is $r=\frac{C}{F}$.

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

## Businesses

### Bond Interest Rate and Spot Price

When the bond price goes up the interest rate goes down, and vice versa. If a bond matures at a price \$ 110, initial purchase price is \$ 100; the interest rate is 10%. If the initial purchase price is $105, the interest rate is lower than 5%.

When bonds are in high demand, the spot price rises, and the interest rate drops. Central bank can bulk purchase bonds so that interest rates are low, and companies and individuals are motivated to spend money rather than purchasing bonds as savings.

## Bond Trading Platforms

* CFETS (China Foreign Exchange Trade System) is a major global trading platform and pricing center for RMB and related products.

## Flat Bond

When a bond does not yield any interest.

* Flat Trading

A situation in which a market or security is neither rising nor declining in price or valuation.

* Default

* Payment-in-kind

Payment-in-kind (PIK) is the use of a good or service as payment instead of cash.
