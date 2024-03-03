# Quantitative Trading Basics

## Performance Measurement

$$
y = \alpha + \beta x + u
$$

where

* $y$ is the total yield
* $\alpha$ is alpha, which is the excess return of the stock or fund.
* $\beta$ is beta, which is volatility relative to the benchmark.
* $x$ is the performance of the benchmark, typical Shanghai-A index for China or S&P 500 index for the U.S.
* $u$ is the residual, which is the unexplained random portion of performance in a given year.

### Key Metrics of A Stock

* Dividend Yield 分红
* (Dividend) Payout Ratio 股息率/派息率: $\frac{\text{TotalDividends}}{\text{NetIncome}}$
* Price-to-Earnings Ratio, P/E ratio 市盈率: $\frac{\text{SharePrice}}{\text{EarningsPerShare}}$, lower the better
* Price-to-Book Ratio, P/B ratio 市净率/市帳率/股價淨值比: $\frac{\text{CompanyTotalShareValue}}{\text{BookValueOfEquity}}=\frac{\text{SharePrice}}{\text{BookValuePerShare}}$, where book value of equity (BVE) is computed by $\text{BVE}=\text{TotalAsset}-\text{TotalLiability}$; P/B ratio is lower the better

For example, as of 2024-02-21 APPLE (NAS: AAPL) dividend yield is \$0.24 per share quarterly. APPLE one share price is \$182, that gives dividend payout ratio $0.52\% \approx \frac{0.24 \times 4}{182}$.
In Dec 2023, APPLE recorded quarterly earnings per share \$2.18 and book value per share \$4.79. Given 2024-02-21 share price \$182, P/E ratio is $20.9 \approx \frac{182}{2.18 \times 4}$, and P/B ratio is $37.97\approx \frac{182}{4.79}$.

### Sharpe Ratio

The Sharpe ratio can be used to evaluate a portfolio's risk-adjusted performance.

$$
Sharpe Ratio =
\frac{R_p-R_f}{\sigma_p}
$$

where

* $R_p$ is return of portfolio
* $R_f$ is risk-free rate
* $\sigma_p$ is the standard deviation of the portfolio's excess return

If Sharpe ratio is smaller than $1$, the portfolio's excess return is relatively small compared to its risk $\sigma_p$, not worthy of studying. If Sharpe ratio is greater than $3$, it is a good investment for high return relative to its risk.

## Common order param explains

### Order Type

* Limit Order

An order submitted with a specified limit price and to be executed at the specified price or better price.

* Market Order

An order submitted without the specified limit price and to be executed against the best bid or the best offer in order.

### Validity Periods and Conditions

* Good for Day (GFD)

Valid until the end of the Day Session of the day (or, until the end of the Night Session if the order is submitted at the Night Session.).

* Good till Date/ Good till Cancel (GTD/GTC)

Valid until the end of the Day Session on the date the specified period ends.
Selectable from either GTD (valid until the end of the day session on the date the specified period ends) or GTC (valid until the cancellation. (If not cancelled, it is valid until the end of the day session on the last trading day.)).

* Fill and Kill (FAK)

In the case where there is unfilled volume after the order is partially executed, cancel the unfilled volume.

* Fill or Kill (FOK)

In the case where all the volume is not executed immediately, cancel all the volume.

### Stop Conditions

* One-Cancels-the-Other Order (OCO)

When either the stop or limit price is reached and the order is executed, the other order is automatically canceled. Define a upper limit and bottom limit to control the fluctuation of price with a frame; if either upper or bottom price is reached and stopped, the opposite limit is canceled.

* Upper/Bottom Limit Trigger Stop

Simple stop (force sell) when reaching a limit.

## Common Trading Strategies

### Leg

A leg is one piece of a multi-part trade, often a derivatives trading strategy, in which a trader combines multiple options or futures contracts, to hedge a position, to benefit from arbitrage, or to profit from a spread widening or tightening. 

When entering into a multi-leg position, it is known as "legging-in" to the trade. Exiting such a position, meanwhile, is called "legging-out". Note that the cash flows exchanged in a swap contract may also be referred to as legs.