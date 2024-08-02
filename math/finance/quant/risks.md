# Risks

## Mark-To-Market (MTM) Risks

MTM risks refer to a group of risk concepts associated with the current market conditions.

It is an accounting practice used to value assets and liabilities given current market conditions; MTM value is computed by $\text{ThisTimePrice} \times \text{NumberOfUnits}$.

In implementation, it includes costs of money, e.g., interests paid to lenders.
If the collateral is bond, need to consider coupon yields.

### Mark-To-Market (MTM) Value as Fair Value

MTM value is the real time price of an asset, and often considered as a close approximation of the fair value of this asset.
However, MTM value is subject to short-time fluctuation that might not reflect the fair value.

Remediation is to include fluctuations (e.g., std variances) to indicate how much MTM prices deviate from moving average price (highly volatile asset MTM prices are not accurate to estimate fair value of the asset).

An alternative is to use moving average as fair value.

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

### Tail Risk/Market Risk (黑天鹅事件风险/小概率事件风险)

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
