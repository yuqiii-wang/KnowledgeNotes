# Repurchase Agreement (REPO, 回购/逆回购)

A repurchase agreement, also known as a repo, RP, or sale and repurchase agreement, is a form of short-term borrowing, mainly in government securities or good-credit corporate bonds.

The dealer sells the underlying security to investors and, by agreement between the two parties, buys them back shortly afterwards, usually the following day, at a slightly higher price (overnight interest).

## REPO vs Reverse REPO

To the party selling the security with the agreement to buy it back, it is a repurchase agreement (REPO).

To the party buying the security and agreeing to sell it back, it is a reverse repurchase agreement (Reverse REPO).

## Repo Types by Management and Ownership

Different underlying management and ownerships have implications of coupon payment to what party, permitted/forbidden re-hypothecation (reuse), what governing legal agreements to use (e.g., what netting methods are allowed), etc.

In this context,

* Buyer: to buy/borrow securities, give out money as interest
* Seller: to sell/lend securities, receive interest as money

### REPO Types By Custody Management

||Ownership|Coupon|Legal Agreement|Re-hypothecation|Comments/Key Characteristics|
|-|-|-|-|-|-|
|Classic/Outright Repo|Full ownership transfer|Retained by Buyer|MRA or GMRA|Allowed (at buyer's discretion)|Also referred as US-style REPO; most typical REPO; to repurchase at a specified price and on a specified date in the future.|
|Tri-Party/Agency Repo|Managed by tri-party agent|Facilitated by Agent|Tri-Party Agreement|Restricted|Collateral is held in an independent third-party account.|
|Held-In-Custody (HIC) Repo|Temporary to buyer in name only, usually favorable to seller|Retained by Seller|Informal custom agreement|Restricted|The seller receives cash for the sale of the security, but holds the security in a custodial account (might not immediately accessible to the buyer) for the buyer.|

### REPO Types By Ownership

||Ownership|Coupon|Legal Agreement|Re-hypothecation|Comments/Key Characteristics|
|-|-|-|-|-|-|
|Classic/Outright Repo|Full ownership transfer|Retained by Buyer|MRA or GMRA|Allowed (at buyer's discretion)|Also referred as US-style REPO; most typical REPO; to repurchase at a specified price and on a specified date in the future.|
|Pledge-Style Repo|Collateral by pledge, not full ownership|Retained by Seller|Informal custom agreement|Restricted|Classic REPO but underlying by pledge|
|Sell/Buy-Back|Full ownership transfer|Included when purchased, e.g., by bond dirty price|Informal custom agreement|Allowed (at buyer's discretion)|Two separate trades: spot sale and forward purchase.|

where in pledge vs outright, although in the scenario of credit default that a seller fails to redeem the lent securities and buyer can exercise the right of disposing/selling the securities, buyer might still need to take legal procedure.
This is worrying and time-consuming in contrast to outright REPO.

Reference:

https://www.boc.cn/en/cbservice/cb2/cb24/200807/t20080710_1320894.html#:~:text=It%20means%20a%20financing%20activity,bonds%20to%20the%20repo%20party

## Repo Types/Products by Tenor and Business Value

"By business value" aims to give an overview of tailored REPO classification by regulation and custom business needs.
In other words, different companies/financial institutions provide different repo products (different tenors/renewal agreements) on markets.

Very often, REPO risks arise from tenor (how long to borrow/lend) and quality of underlying securities (high/low quality bonds).
As a result, REPO products are regulated/designed per these two aspects.

### Repo Type/Product Examples Stipulated by Authority

Available REPO tenors depend on regulations per country.
For example, by 2024, Shenzhen Stock Exchange (SZSE, 深圳证券交易所) states only such tenors are allowed in REPO trading.

|Underlying Bonds|Available Tenors|
|-|-|
|High-quality bonds, e.g., gov bonds|1,2,3,4,7,14,28,63,91,182,273 days|
|Lower-quality bonds, e.g., corp bonds|1,2,3,4,7 days|

### Repo Type/Product Examples Provided by A Financial Institution

A financial institution can provide tailored REPO agreements to REPO counterparties for trading (of course under authority regulations), giving them flexibility of how to pay coupon, lending/borrowing auto-renewal, purchase with different currencies, etc.

||Governing Document|Underlying Asset|Tenor|Legal Title Transfer|Margining|Business|
|-|-|-|-|-|-|-|
|Typical Repo/Reverse Repo|GMRA|Gov bonds, Credit, Equity|Open, Overnight ~ 5 years|Yes|Daily|Repo: Deploy bond/equity for cash borrow; Reverse repo: Collateralized cash lending for interest|
|Cross Currency Repo|GMRA|Gov bonds, Credit|Open, Overnight ~ 5 years|Yes|Daily|Mostly USD funding to meet US reserve requirements|
|Extendible/Rollover Repo|GMRA|Gov bonds, Credit, Equity|3 month start with 1 month increment, 1 year start with 1 month increment, etc|Yes|Daily|Extendible agreement periods, typically every 3 months to renew repo termination date|
|Evergreen Repo|GMRA|Gov bonds, Credit, Equity|Notice period > 30 days|Yes|Daily|Even greater flexible to renew repo termination date|
|Total Return Swap|ISDA/CSA|Gov bonds, Credit, Equity|Open, Overnight ~ 1 years|No|Daily|By Repo + Credit Default Swap (CDS) as the reference asset, borrower can leverage much more money by only paying interest to repo and premium to reference asset third party agency|
|Bond Forward|ISDA/CSA|Gov bonds, Credit|Overnight ~ 8 years|Yes|Daily|Bonds are agreed to be sold at a pre-determined price in the future/mark-to-market value at the forward date. This can mitigate the risk associated with bond price volatility between spot and forward dates. |
|Unsecured Bond Borrowing|GMRA|Gov bonds, Credit, Equity|Overnight ~ 1 years|No|None|Borrower borrows money without providing collateral; they are charged high repo interests|

#### Global Master Repurchase Agreement (GMRA)

Global Master Repurchase Agreement (GMRA) published by International Capital Market Association (ICMA) is standardized legal agreement used globally for repurchase agreements (repos).

The GMRA covers classic repos, sell/buy-backs, and other similar transactions.

Key Objectives of the GMRA

* Provide a standardized framework for repo transactions.
* Reduce legal and operational risks by clarifying the rights and obligations of both parties.
* Facilitate netting and close-out in the event of default.

#### Credit Support Annex (CSA)

A Credit Support Annex (CSA) is a document that defines the terms for the provision of collateral by the parties in derivatives transactions, developed by the International Swaps and Derivatives Association (ISDA).

## REPO RFQ (报价咨询)

RFQ (Requests For Quotes) is for repo sales/traders asking counterparty sales/traders for his/her quotes for REPO trade and the underlying securities.

For example, a sales asks a counterparty trader/sales:

```txt
Show your bid on ISIN US12345678JK USD 5mm for 2mth and 3mth
```

The counterparty trader/sales replies:

```txt
My bid for US12345678JK:
5mm 2mth 5hc 3.5%
5mm 3mth 5hc 3.7%
```

that the RFQ requester wants to sell (requester offer <---> responser bid) USD `5mm` (5 million) worth of `US12345678JK` by dirty price.

The reply shows the trader/sales is willing to bid with haircut discount by `5%` and paid interest rate of `3.5%` and `3.7%` for borrowing `US12345678JK` for 2 months and 3 months.
The longer the borrowing period, the higher risk of the repo trade, hence the higher interest rate the lending party might charge.

### REPO RFQ Types

#### Outright REPO 买断式回购

The seller (borrower) **transfers ownership** of the securities to the buyer (lender) with an agreement to repurchase them at **a specified price/rate and on a specified date** in the future.

Implications: they can sell or reuse the securities (e.g., for further collateralization or trading).

#### AON (All-or-None, 全部或零)

AON (All-or-None) is a trading condition where an order must be executed in its **entirety or not at all**.

If the entire order cannot be filled at the specified price or quantity, the order is canceled.

## Repo Market Players

* Investors

Cash-rich institutions; banks and building societies

* Borrowers

Traders; financing bond positions, etc

* Tri-Party

Tri-Party is a third-party agent (the tri-party agent) intermediates between two primary parties: the collateral provider (borrower) and the collateral receiver (lender).

Tri-party agent can help increase operational efficiency and flexibility, improve liquidity, and mitigate default risks, for collateral allocation and management, settlement and custody, valuation and margining services are provided by the tri-party agent.

Popular try-party agents are

-> Euroclear (A leading provider of triparty services in Europe)

-> Clearstream

## Business Motivations

### Central Bank Control over Money Supply

Central bank is rich in cash and has a large reserve of national bonds, so that it can influence the markets.

* Injecting Liquidity (Reverse Repo):

Central bank buys securities from commercial banks with the promise to sell them back at a predetermined date and price.
Commercial banks receive money then by loan to distribute money to the society.

* Absorbing Liquidity (Repo)

Central bank sells securities to the banks with an agreement to buy them back later.

#### Examples

* The United States Federal Reserve Used Repo for Federal Funds Rate (FFR) Adjustment

Repurchase agreements add reserves to the banking system and then after a specified period of time withdraw them;
reverse repos initially drain reserves and later add them back.
This tool can also be used to stabilize interest rates (FFR).

* Relationship Between Repo and SOFR

Secured Overnight Financing Rate (SOFR) is a broad measure of the cost of borrowing cash overnight collateralized by Treasury securities with a diverse set of borrowers and lenders;.
It is based entirely on transactions (not estimates), hence serving a good alternative to London Inter-Bank Offered Rate (LIBOR).

SOFR reflects transactions in the Treasury repo market, e.g., UST (U.S. Treasury), that in the Treasury repo market, people borrow money using Treasury debt as collateral.

* Central bank uses reverse repo to buy securities from financial institutions (typically state-owned banks) to increase their capital reserve/liquidity

SHANGHAI, Nov 29, 2024 (Reuters) - China's central bank said on Friday it conducted 800 billion yuan ($110.59 billion) of outright reverse repos in November.

The People's Bank of China (PBOC) said the repo operations aimed to keep liquidity in the banking system adequate at a reasonable level.

This happened for real estate saw sharp drops in prices that resulted in banks losing money.
Feared of banks went bankrupt, PBOC decided to inject temporary liquidity into banks.

### Individual Financing Needs
