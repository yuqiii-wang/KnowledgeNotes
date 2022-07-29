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