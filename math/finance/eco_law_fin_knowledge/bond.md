# Bond

* Zero coupon bonds

A zero-coupon bond, also known as an accrual bond, is a debt security that does not pay interest but instead trades at a deep discount, rendering a profit at maturity, when the bond is redeemed for its full face value.

* Step-up bonds

A step-up bond is a bond that pays a lower initial interest rate but includes a feature that allows for rate increases at periodic intervals.

* Deferred interest bonds

A deferred interest bond, also called a deferred coupon bond, is a debt instrument that pays all of its interest that has accrued in the form of a single payment made at a later date rather than in periodic increments.

* Coupon Factor

The Factor to be used when determining the amount of interest paid by the issuer on coupon payment dates.

* Coupon Rate

The interest rate on the security or loan-type agreement, e.g., $5.25\%$. In the formulas this would be expressed as $0.0525$.

* Day Count Factor

Figure representing the amount of the Coupon Rate to apply in calculating Interest. 

* Pool factor

A pool factor is the outstanding principle out of the amount of the initial principal for ABS or MBS.

$$
F_{pool} = \frac{OutstandingPrincipleBalance}{OriginalPrincipleBalance}
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
* 30/360: U.S. corporate and mmunicipal bonds
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

## Bond valuation

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
\big)
+
\frac{F}{(1+r)^n}
$$
where coupon interest rate is $r=\frac{C}{F}$.


## Terminologies

* Interlife

Any action in the life cycle (between start/end date) of this repo.

* Accrued interest

Accrued interest is the interest on a bond or loan that has accumulated since the principal investment, or since the previous coupon payment if there has been one already.

In other words, the interest accounts for the time since the bond's start date or the last coupon payment date. 

* Clean and dirty price

"Clean price" is the price excluding any interest that has accrued.

"Dirty price" (or "full price" or "all in price" or "Cash price") includes accrued interest

* Matched book

A matched book is an approach that banks and other institutions may take to ensure that the maturities of its assets and liabilities are equally distributed. 
A matched book is also known as "asset/liability management" or "cash matching."

* Bullet Loan vs Amortizing Loan

A typical amortizing loan schedule requires the gradual repayment of the loan principal over the borrowing term. 
However, a bullet loan requires one lump sum repayment of the loan principal on the date of the maturity.

## Businesses

### Bond Interest Rate and Spot Price

When the bond price goes up the interest rate goes down, and vice versa. If a bond matures at a price $110, initial purchase price is $100; the interest rate is 10%. If the initial purchase price is $105, the interest rate is lower than 5%.

When bonds are in high demand, the spot price rises, and the interest rate drops. Central bank can bulk purchase bonds so that interest rates are low, and companies and individuals are motivated to spend money rather than purchasing bonds as savings.

### Convertible Bond

A convertible bond or convertible note or convertible debt (or a convertible debenture if it has a maturity of greater than 10 years) is a type of bond that the holder can convert into a specified number of shares of common stock in the issuing company or cash of equal value.

### Negotiable Certificate of Deposit (NCD)

NCDs are guaranteed by banks and can usually be sold in a highly liquid secondary market, 
but they cannot be cashed in before maturity.

In other words, it is a type of bond issued by bank.

Interest rates are negotiable, usually paid either twice a year or at maturity, or the instrument is purchased at a discount to its face value.

## Bond Trading Platforms

* CFETS (China Foreign Exchange Trade System) is a major global trading platform and pricing center for RMB and related products.