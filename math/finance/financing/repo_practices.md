# Repo Practices

## Repo Types by Collateral Ownership

* Classic Repo/Specified Delivery

Required the delivery of a pre-specified bond at the onset, and at maturity of the contractual period. 

One party "sells" bonds to the other while simultaneously agreeing to repurchase them or receive them back at a specified future date.

Also known as allocation trade for bonds as the underlying security should be allocated.

* Held-In-Custody (HIC) Repo

In a held-in-custody repo, the seller receives cash for the sale of the security, but holds the security in a custodial account (might not immediately accessible to the buyer) for the buyer. 
This type of agreement is even less common because there is a risk the seller may become insolvent and the borrower may not have access to the collateral.

* Tri-Party Repo

Basically a basket form of transaction and allows for a wider range of instruments in the basket or pool.

Tri-party repo is a type of repo contract where a third entity (apart from the borrower and lender), called a Tri-Party Agent, acts as an intermediary between the two parties to the repo to facilitate services like collateral selection, payment and settlement, custody and management during the life of the transaction.

Collateral is held in an independent third-party account.

The tri-party agent is also custodian, manages exchange of collateral and cash internally

Tri-party repo rate is usually higher than the delivery repo rate, but lower than HIC repo.

## Repo Types by Duration and Business Value

||Governing Document|Underlying Asset|Tenor|Legal Title Transfer|Margining|Business|
|-|-|-|-|-|-|-|
|Typical Repo/Reverse Repo|GMRA|Government, Credit, Equity|Open, Overnight ~ 5 years|Yes|Daily|Repo: Deploy bond/equity for cash borrow; Reverse repo: Collateralized cash lending for interest|
|Cross Currency Repo|GMRA|Government, Credit|Open, Overnight ~ 5 years|Yes|Daily|Mostly USD funding to meet US reserve requirements|
|Extendible Repo|GMRA|Government, Credit, Equity|3 month start with 1 month increment, 1 year start with 1 month increment, etc|Yes|Daily|Extendible agreement periods, typically every 3 months to renew repo termination date|
|Evergreen Repo|GMRA|Government, Credit, Equity|Notice period > 30 days|Yes|Daily|Even greater flexible to renew repo termination date|
|Triparty Repo|GMRA|Government, Credit, Equity|Open, Overnight ~ 5 years|Yes|Daily|Triparty agency handles trade and collateral operations, provided additional trust/credit|
|Total Return Swap|ISDA/CSA|Government, Credit, Equity|Open, Overnight ~ 1 years|No|Daily|By Repo + Credit Default Swap (CDS) as the reference asset, borrower can leverage much more money by only paying interest to repo and premium to reference asset third party agency|
|Bond Forward|ISDA/CSA|Government, Credit|Overnight ~ 8 years|Yes|Daily|Bonds are agreed to be sold at a pre-determined price in the future/mark-to-market value at the forward date. This can mitigate the risk associated with bond price volatility between spot and forward dates. |
|Unsecured Bond Borrowing|GMRA|Government, Credit, Equity|Overnight ~ 1 years|No|None|Borrower borrows money without providing collateral; they are charged high repo interests|

* Document Explained

Global Master Repurchase Agreement (GMRA) is the principal master agreement for cross-border repos globally, as well as for many domestic repo markets containing standard provisions such as minimum delivery periods.

A Credit Support Annex (CSA) is a document that defines the terms for the provision of collateral by the parties in derivatives transactions, developed by the International Swaps and Derivatives Association (ISDA).

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

* The United States Federal Reserve Used Repo for Federal Funds Rate (FFR) Adjustment

Repurchase agreements add reserves to the banking system and then after a specified period of time withdraw them; 
reverse repos initially drain reserves and later add them back. 
This tool can also be used to stabilize interest rates (FFR).

* Relationship Between Repo and SOFR

Secured Overnight Financing Rate (SOFR) is a broad measure of the cost of borrowing cash overnight collateralized by Treasury securities with a diverse set of borrowers and lenders;.
It is based entirely on transactions (not estimates), hence serving a good alternative to London Inter-Bank Offered Rate (LIBOR).

SOFR reflects transactions in the Treasury repo market, e.g., UST (U.S. Treasury), that in the Treasury repo market, people borrow money using Treasury debt as collateral.

* The United States Federal Overnight Reverse Repo (O/N RRP)

This is a cash liquidity management tool that attracts depository institutions to place money in the Fed who pays interests to the depository institutions, so that less funds are being lent in the market.

The Federal Reserve manages *overnight interest rates* (the rates at which banks lend funds to each other at the end of the day in the overnight market to meet federally-mandated reserve requirements) by setting the *interest on reserve balances* (IORB) rate, which is the rate paid to depository institutions (e.g., banks) on balances maintained at Federal Reserve Banks.

Every day, the Federal Reserve accepts overnight cash investments from banks, government-sponsored enterprises (the housing agencies plus the Federal Home Loan Banks), and money market mutual funds and provides Treasury securities (gov bonds) as collateral at its Overnight Reverse Repurchase Agreement (ON RRP) facility.

The overnight means the repo has a duration of one night.

## Risks

* Counterparty / Credit Risk

Counterparty risk is risk of default due to financial difficulty or withdrawal from business
Banks internally rate all counterparties and assign exposure limits to each, by firm and sector

* Collateral risk / Issuer risk

Quality of collateral held suffering due to decline in fortunes of issuer; 
lower grade collateral trades at a higher spread to government repo rate

* Market Risk

Risk exposure from changes in market levels, interest rates, asset values, etc. 
One reason for continuing popularity in stock lending

* FX risk

Cross-currency repo, or a stock loan collateralized with
assets denominated in a different currency

### Risk Rating Table

||S&P|Moody's|
|-|-|-|
|Top Quality|AAA|Aaa|
|High Quality|AA+|Aa1|
||AA|Aa2|
||AA-|Aa1|
|Upper Medium|A+|A1|
||A|A2|
||A-|A1|
|Medium|BBB+|Baa1|
||BBB|Baa2|
||BBB-|Baa3|

## Trading and Hedging Strategies

* Yield Curve Arbitrage

Anticipated that some securities' price might go up; some might go down.
Buy both so that future yield would be neutral to avoid drastic rises or falls in values.

* Credit Intermediation

Gov bonds usually have high credibility.
They can be used to hedge risks.

Typically, use SOFR (for USD treasury bond) or LIBOR (for British treasury bond) as the benchmark to measure the risk of a bond. 
A haircut/spread/added floating rate can be applied on top of SOFR or LIBOR as the risk hedging quantified strategy.

* Matched Book Trading

Make sure underlying securities have its value as stated by daily monitoring.

Counterparties are monitored as well. 

Consider *Prime*: prime bonds/trades/clients are private agreements that establish good trust-worthy relationships with the counterparties.
The prime agreements offer good repo rate to counterparties. 

* Derivative Market Anticipation

Demand for short-term cash is often correlated to the derivative market.
High volatility market means that people are rush to raise funds to short/long derivatives.
