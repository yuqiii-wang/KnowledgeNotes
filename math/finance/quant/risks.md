# Risks

## Mark-To-Market (MTM) Risks

MTM risks refer to a group of risk concepts associated with the current market conditions.

It is an accounting practice used to value assets and liabilities given current market conditions; MTM value is computed by $\text{ThisTimePrice} \times \text{NumberOfUnits}$.

In implementation, it includes costs of money, e.g., interests paid to lenders.
If the collateral is bond, need to consider coupon yields.

### Margin-at-Risk (MaR)

Margin-at-Risk (short: MaR) quantifies the "worst case" margin-call and is only driven by market prices （持有仓位低于保证金）.

Margin itself refers to how much left between loaned money vs current collateral value.

### Liquidity Risk （流通风险）

Liquidity risk talks about if an asset is easy to sell in market.

To quantify liquidity, one can compute recent traded volumes and number of counterparties.

### Value-at-Risk (VaR) （最大可能损失预估）

For a given portfolio, time horizon, and probability $p$, the $p$ VaR can be defined informally as the maximum possible loss during that time after excluding all worse outcomes whose combined probability is at most $p$.
This assumes mark-to-market pricing, and no trading in the portfolio.

Factors are

* forecast of foreign exchange for this currency
* forecast of treasury bond interests
* history price fluctuationFrecords of this kind of security given various market conditions

### Tail Risk/Market Risk

In short, it means sudden change of market conditions.

Tail risk is the financial risk of an asset or portfolio of assets moving more than three standard deviations (likelihood of happening $1-P(\mu-3\sigma \le X \le \mu + 3\sigma)\approx 1-99.7\%=0.3\%$) from its current price, above the risk of a normal distribution.

It is used to refer the risk associated with an event (black swan) that is extremely unlikely to happen.

## Liquidity-at-Risk (LaR)

Liquidity-at-Risk (short: LaR, not confused with liquidity risk) of a financial portfolio associated with a stress scenario is the net liquidity outflow resulting from this stress scenario:

Liquidity at Risk =
Maturing Liabilities (repayment to borrowed money) 
$+$ Net Scheduled Outflows (some planned payment to others) 
$+$ Net Outflow of Variation Margin (short-term market fluctuations) 
$+$ Credit-Contingent Cash Outflows (investor withdrawal when financial institutions have low credit)

If the Liquidity at Risk is greater than the portfolio's current liquidity position then the portfolio may face a liquidity shortfall.

### Stress Test

A stress test is an analysis or simulation designed to determine the ability of a given financial instrument or financial institution to deal with an economic crisis.

Gov plans for scenarios such as sudden drops of employment rate, interest rate rises, etc.

## Short-Term Interbank Exchange Rate as A Risk Indicator

Short-Term interbank exchange rate refers to interest rate of borrowing money between major banks to meet regulatory requirements, e.g., minimum reserve a bank must hold overnight.

Short terms typically talk about overnight (spot next (S/N)), one week, one month, two months, three months, six months, and 12 months.

Interbank rates are referred as below names in different jurisdictions.

* London Inter-Bank Offered Rate (LIBOR)
* Euro Overnight Index Average (EONIA)
* Secured Overnight Financing Rate (SOFR) 
* Hong Kong Interbank Offered Rate (HIBOR)
* China Interbank Interest Rate

Interbank interest rate is almost as safe as treasury bonds, hence considered as money cost baseline.

Yield curves (interests from lending/borrowing money) of such rates have below implications

* Upward sloping: long-term yields are higher than short-term yields. Healthy mode, economy is expanding
* Downward sloping: short-term yields are higher than long-term yields. Unhealthy mode, economy is in recession. Short-term rate is low indicating little needs of money/low liquidity.
* Flat: economy forecast is uncertain.

## Discount Security Risk Analysis by NPV

*Present value* (PV) is defined as

$$
\begin{align*}
    & \text{exponential form }
    && PV = \frac{C_n}{(1+r)^n} \\
    & \text{linear form }
    && PV = \frac{C_n}{1+rn}
\end{align*}
$$

where

* $C_n$ cash flow at the $t$ time
* $r$ discount rate/IRR (Internal Risk of Return)
* $n$ the number of time periods

*Net Present Value* (NPV) is computed by measuring the difference between present cash (PC) flow and cost of cash (CC) over $N$ periods.

$$
\begin{align*}
    & \text{exponential form }
    && \text{NPV} \sum_{n=1}^{N} \frac{PC_n}{(1+r)^n} - \sum_{n=1}^{N} \frac{CC_n}{(1+r)^n} \\
    & \text{linear form }
    && \text{NPV} \sum_{n=1}^{N} \frac{PC_n}{1+rn} - \sum_{n=1}^{N} \frac{CC_n}{1+rn} \\
\end{align*}
$$

The NPV says if $NPV>0$, there is surplus in cash flow (profit); if $NPV<0$, there is loss in cash flow.

NPV can be used as a risk indicator before a security (typically bonds) mature.

In practice, rather than using one NPV targeting one particular date, risks are mapped to the maturity date's nearest tenor.

|Tenor $n$|Rate|
|-|-|
|$\text{o}/\text{n}$ (overnight)|$r_0$|
|$\text{t}/\text{n}$ (tomorrow)|$r_1$|
|2 days|$r_2$|
|5 days|$r_5$|
|1 week|$r_7$|
|2 weeks|$r_{14}$|
|...|...|
|30 years|$r_{10950}$|

where

* $n$: grid point of period with $n \in \{ 0, 1, 2, 5, 7, ..., 10950 \}$
* $r_n$ interest rate of instrument for period $n$

Set $t$ as the remaining days to maturity, $t_{\text{lower}}$ and $t_{\text{upper}}$ as the lower nearest tenor and upper nearest tenor, e.g., $t=10$ sees $t_{\text{lower}}=n_{\text{1week}}$ and $t_{\text{upper}}=n_{\text{2week}}$.

Define a ratio $\gamma_{\text{lower}}$ and $\gamma_{\text{upper}}=1-\gamma_{\text{lower}}$ that splits NPV as risk to 

$$
\gamma_{\text{lower}} =
\frac{t_{\text{lower}}(t_{\text{upper}}-t)}{t(t_{\text{upper}}-t_{\text{lower}})}
$$

So that, the final tenor-mapped NPVs are
$\text{NPV}_{\text{lower}}=\gamma_{\text{lower}} \text{NPV}$ and $\text{NPV}_{\text{upper}}=\gamma_{\text{upper}} \text{NPV}$.

### Example

For example, a one-year bond's spot price is \$102.28, and this bond has 10 days to mature paying \$100 + \$2.3 (assumed annual payment).

The present value of the bond is \$$102.24 = 100 + 2.3  \times \frac{355}{365}$.
The NPV is \$$-0.04 = 102.24 - 102.28$ that indicates possible loss of \$0.04 per \$100 when this bond matures.
$\text{NPV} < 0$ indicates this bond is likely over-priced. A further explanation can be that market is in turmoil and money seeks refuge by investing in good quality bonds.

To map the NPV risk to its nearest tenors, there is

$$
\begin{align*}
\gamma_{\text{lower}} &=
\frac{t_{\text{lower}}(t_{\text{upper}}-t)}{t(t_{\text{upper}}-t_{\text{lower}})} =
\frac{7 \times (14 - 10)}{10 \times (14 - 7)} = 0.4 \\
\gamma_{\text{upper}} &= 1 - \gamma_{\text{lower}} = 0.6
\end{align*}
$$

In conclusion, $\text{NPV}_{\text{lower}}=-0.016=-0.04 \times 0.4$ and $\text{NPV}_{\text{lower}}=-0.024=-0.04 \times 0.6$.

### Bond-Based NPV and Risks

In bonds, $\text{SecurityPrice}$ is used as the spot price of the bond (present value).

$$
\text{SecurityPrice} =
\text{SecurityMidPrice} \times \text{CurrencyExchange} \times \text{IssueFactor}
$$

where $\text{SecurityMidPrice} = \frac{1}{2}(\text{SecurityAskPrice} + \text{SecurityBidPrice})$.
The $\text{IssueFactor}$ refers to partial purchase of the total amount of the issued bonds.
$\text{CurrencyExchange}$ is often used to convert bond value to US dollars.

$\text{StartCash}$ represents cost of cash.

$$
\text{StartCash} =
(\text{SettlementPrice} / 100) \times \text{Quantity} \times \text{HaircutRatio}
$$

where $\text{SettlementPrice}$ is a bond face value (typical \$100 per coupon).

The NPV as risk is $\frac{1}{(1+r)^n}\text{SecurityPrice}-\text{StartCash}$, where $r$ is coupon rate and $n$ is the total number of coupon yields.