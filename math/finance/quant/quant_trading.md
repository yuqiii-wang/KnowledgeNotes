# Quantitative Trading Basics

## Performance measurement

$$
y = \alpha + \beta x + u
$$
where
* $y$ is the total yield
* $\alpha$ is alpha, which is the excess return of the stock or fund.
* $\beta$ is beta, which is volatility relative to the benchmark.
* $x$ is the performance of the benchmark, typical Shanghai-A index for China or S&P 500 index for the U.S.
* $u$ is the residual, which is the unexplained random portion of performance in a given year. 

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

*  Fill or Kill (FOK) 	

In the case where all the volume is not executed immediately, cancel all the volume.

### Stop Conditions

* One-Cancels-the-Other Order (OCO) 

When either the stop or limit price is reached and the order is executed, the other order is automatically canceled. Define a upper limit and bottom limit to control the fluctuation of price with a frame; if either upper or bottom price is reached and stopped, the opposite limit is canceled.

* Upper/Bottom Limit Trigger Stop

Simple stop (force sell) when reaching a limit.