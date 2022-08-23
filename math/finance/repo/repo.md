# Repurchase Agreement

A repurchase agreement, also known as a repo, RP, or sale and repurchase agreement, is a form of short-term borrowing, mainly in government securities.

The dealer sells the underlying security to investors and, by agreement between the two parties, buys them back shortly afterwards, usually the following day, at a slightly higher price (overnight interest).

### Motivations

This instrument is used in speculative trading (high-medium level frequency trading) to finance purchasing (by long/short) a product. Since repo often needs to repay within a few days, it is cheap in paying interest and flexible in arranging buy back dates.

Another use case is satisfying regulatory requirements that banks are required to preserve some cash. A seller of a repo, such as an investment bank, borrows money from reserve banks to meet certain short term auditory requirements, hands over owned federal bonds as collaterals to the reserve bank.

Federal Reserve uses it as a monetary policy tool. When the Fed buys securities from a seller who agrees to repurchase them, it is injecting reserves into the financial system. Conversely, when the Fed sells securities with an agreement to repurchase, it is draining reserves from the system. 

## Reverse repo

To the party selling the security with the agreement to buy it back, it is a repurchase agreement. 

To the party buying the security and agreeing to sell it back, it is a reverse repurchase agreement. The reverse repo is the final step in the repurchase agreement, closing the contract.

## Initial margin (Haircut)

Initial margin is the excess of cash over securities or securities over cash in a repo or securities at the market price when lending a transaction.

Haircut serves as a discount factor to reflect risk aversion consideration of the bond price fluctuations during the lent period. Hence, the actual lent money is slightly lower than the bond market value.

|Haircut method|Haircut formula|
|-|-|
|Divide|$Haircut=100\times \frac{CollateralMktValue}{CashLent}$|
|Multiply|$Haircut=100\times \frac{CashLent}{CollateralMktValue}$|

## Repo vs Sell/Buy Back

|Repo|Sell/Buy Back|
|-|-|
|Coupon rate is not considered, that seller/original owner of the bond can still receive full interest payment.|Coupon rate is considered, that seller/original owner of the bond needs to hand over the interest payment during the particular period of lending time to its buyer|
|On coupon payment date, the coupon should be temporarily handed over to its seller|Coupon interest payment is incorporated into forward price already.|

## Example

### Repo calculation

* Trade date: 20th Jul 2014
* Trade price: $110.85 (per $100 coupon)
* Face value: $10,000,000
* Security type: 12.5% interest rate bond
* Last coupon date: 1st Jul 2014
* Repo rate: 7.5%
* repo term: 4 days (implies a return date of 24th Jul 2014)
* Day count conversion: Actual/360
* Haircut: 102
* Haircut method: divide

Solution:
1. First leg:

On 20th Jul 2014, the seller of the bond repo receives the borrowing money and gives his bond as collateral to the buyer.

Last coupon date refers to the last date the coupon interest is materialized, so that accrued interest for next time payment should start from the last coupon date. 

The time when this coupon is purchased, the accrued interest should be added.

par value of this repo: 
$110.85 \times 10,000,000/100 = 11,085,000$

Accrued interest: $12.5\% \times 10,000,000 \times 19/360 = 65,972.22$

Start cash/original borrowing before haircut: $11,085,000+65,972.22=11,150,972.22$

Start cash/original borrowing: $11,150,972.22 \div \frac{102}{100} = 10,932,325.71$

2. Second leg:

On 24th Jul 2014, repo is returned to the seller who should pay the borrowing money plus repo interest to the repo's buyer.

Repo interest: $7.5\% \times 10,932,325.71 \times 4/360 = 9,110.27$

End Cash: $10,932,325.71 + 9,110.27 = 10,941,435.98$
