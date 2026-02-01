# Financing Interest Rates

In the financing market, different market participants (PBOC/banks) and tenors have different market implications, and below explain the reasons.

## The Complete China Financing Matrix (by 2025)

| Instrument | Participants | Time Horizon | Primary Role | Mechanism | Determination Method | Impact to Market | Comments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PBOC 7-Day Reverse Repo** | **PBOC $\to$ Banks** | Short (**7 Days**) | **Price Signal** (Anchor) | **Pledged** (Collateralized) | **Fixed Rate, Quantity Bidding**<br>(PBOC sets Price, Banks ask for Amount) | **Sets the Floor.** Defines the cost of policy money. | Central bank anchor |
| **R007**<br>(Interbank Repo) | **Bank $\leftrightarrow$ Non-Bank** | Short (**7 Days**) | **Short-Term Liquidity** | **Pledged** (Collateralized) | **Market Trading.**<br>Weighted Average of all interbank trades. | **Liquidity Ease.** Shows if Non-Banks are stressed. | Tracks PBOC OMO but usually higher (Spread = Credit Risk). |
| **R001**<br>(Interbank Repo) | **Bank $\leftrightarrow$ Non-Bank** | Short (**1 Day**) | **Daily Plumbing** | **Pledged** (Collateralized) | **Market Trading.**<br>Weighted Average of overnight trades. | **Immediate Stress Test.** | **Highest Volume.** ~85% of all repo volume is R001. |
| **GC001**<br>(Exchange Repo) | **Retail/Funds $\leftrightarrow$ Borrowers** | Short (**1 Day**) | **Retail Cash Parking** & Stock Leverage | **Pledged** (Collateralized by Treasuries) | **Continuous Auction.**<br>Order book matching (like a stock ticker). | **Retail Sentiment.** Links stock market to bond market. | |
| **Mid-Term Outright Repo** | **PBOC $\to$ Banks** | Medium (**3 - 6 Months**) | **Bridge** (Liquidity) | **Outright** (Asset Transfer) | **Fixed Amount, Rate Bidding**<br>(PBOC sets Amount, Banks bid Price) | **Liquidity Ease.** Smoothes medium-term friction. | **New Tool (since Oct 2024).** Collateral actually changes hands (unlike Pledged). |
| **MLF**<br>(Medium-term Lending Facility) | **PBOC $\to$ Banks** | Medium/Long (**1 Year**) | **Volume Support** | **Pledged** (Collateralized) | **Fixed Amount, Rate Bidding**<br>(PBOC sets Amount, Banks bid Price) | **Bank Stability.** Ensures banks have long-term liabilities. | Old MLF: Was a "Fixed Rate Tender" (Price fixed, Quantity flexible).<br>New MLF (since 2024): Is now a "Fixed Quantity, Rate Tender" (same as the Outright Repo). |
| **SHIBOR** | **Bank $\leftrightarrow$ Bank** | **Overnight to 1 Year** | **Market Pulse** | **Unsecured** (Trust) | **The "18-Bank Panel."**<br>Trimmed Mean of quotes (Top 4/Bottom 4 excluded). | **Risk Indicator.** Shows if banks trust each other. | |
| **LPR**<br>(Loan Prime Rate) | **Banks $\to$ Economy** | **1 Year** & **5+ Years** | **Borrowing Cost** | **Quotation** (Formula) | **The "LPR Panel."**<br>Funding Cost + Risk Spread. | **Monthly Bills.** | **5-Year LPR** dictates Mortgage rates. |

where

* PBOC stands for People's Bank of China, China's central bank.

### The China Repo Rates

There are parallel repo markets in China

* DR (Depository Repo): e.g., DR007, Only Banks can enter.
* R (General Repo): e.g., R007 Everyone (Banks + Hedge Funds + Securities Firms) can enter. Real time update.
* FR (Fixed Rate Repo): e.g., FR007/FDR007, Snapshot of repo rate, Once a day. Announced at ~11:30 AM.

Since banks can always borrow cheaper in the exclusive "DR" club, they only enter the "R" market to lend money to Non-Banks (who are not allowed in the DR club) to earn a higher interest rate. Therefore, the R007 rate is effectively determined by the supply of cash from Big Banks and the demand for cash from Non-Banks.

### The Market Motivation Matrix

| Instrument | Full Name | Economy Motivation Real World Example |
| :--- | :--- | :--- | 
| **PBOC 7-Day Reverse Repo (OMO)** | **7-Day Reverse Repurchase Agreement** (Open Market Operation) | July 22, 2024, PBOC cut the 7-Day Reverse Repo (OMO) rate from 1.8% to 1.7% (a 10 basis point cut) to stimulate economy. |
| **Outright Reverse Repo** | **Outright Reverse Repurchase Agreement** | Late January 2026, the PBOC utilized the Outright Reverse Repo Facility to inject a massive sum (estimated ~800 Billion Yuan) into the banking system to stabilize economy for before The Lunar New Year. Chinese companies often withdraw massive cash for year-end bonuses, and households for "Hongbao" (red envelopes). This drains liquidity from banks at the fastest rate of the year. |
| **MLF** | **Medium-term Lending Facility** | September 25, 2024 the PBOC cut the MLF rate by a massive 30 basis points (from 2.30% to 2.00%) to revitalize the economy, explicitly decoupling MLF from the LPR to lower bank funding costs. |
| **SHIBOR** | **Shanghai Interbank Offered Rate** | June 20, 2013, SHIBOR rate spiked to an all-time high of 13.44%. The PBOC deliberately withheld liquidity to punish banks for risky "Shadow Banking" practices (Wealth Management Products). Banks stopped trusting each other, leading to a freeze in interbank lending. |
| **LPR** | **Loan Prime Rate** | February 20, 2024, The PBOC cut the 5-Year LPR by 25 basis points (from 4.20% to 3.95%), but no change to 1-year LPR, to save the property market, without just dumping generic cash. |
---

References:

* PBOC cuts key rates to steady growth
    * By SHI JING in Shanghai | China Daily | Updated: 2023-08-16 08:51
    * https://global.chinadaily.com.cn/a/202308/16/WS64dc161fa31035260b81c5e8.html


### Price Determination Mechanism

#### PBOC 7-Day Reverse Repo (OMO): The "Fiat" Decision

PBOC with very large (almost infinite) liquidity directly determines a money lending rate.
Since PBOC is the most trustworthy entity and has very high liquidity, once the rate is set up, it becomes the benchmark that no market force can push the rate higher than this (arbitrage would kill it instantly).
This is why it is the Anchor.

#### MLF & Outright Repo: The "Auction" Mechanism

Only "Primary Dealers" (about 50 select banks) can enter the room.
MLF has been the major tool of PBOC a few years ago, now gradually substituted by mid-term outright (reverse) Repo for mid-term Repo has more efficiency.

||MLF|Repo|
|:---|:---|:---|
|Mechanism|PBOC announces a **fixed rate**, **commercial banks bid for volume**|PBOC announced the **volume** of liquidity and ask **commercial banks to bid rates and volume**, the most favorable rates win the bids|
|Responsiveness|Laggy. Updated once a month.|Real-time. Updated daily/weekly.|
---

#### Non-PBOC Repo: Short-Term Financing Needs

FR007/FDR007 is a typical rate reflecting real world financing needs.
For example in the year end 31 Dec 2025, companies need money to fill bank accounts to make their financial reports look good, and these needs drive repo rate up.

<div style="display: flex; justify-content: center;">
      <img src="imgs/cfets_repo_20260129.png" width="30%" height="60%" alt="cfets_repo_20260129" />
</div>
</br>

#### SHIBOR: The Financing Needs

* Every morning at 10:55 AM, they submit a rate to the CFETS (China Foreign Exchange Trade System) answering: "What rate would you charge to lend unsecured cash to another high-credit bank?"
* The Participants: Includes the "Big 6" State Banks (ICBC, BOC, Construction Bank, AgBank, etc.), Joint-stock banks (CITIC, Minsheng), and a few foreign banks (like HSBC China).

<div style="display: flex; justify-content: center;">
      <img src="imgs/cfets_shibor_20260129.png" width="30%" height="60%" alt="cfets_shibor_20260129" />
</div>
</br>

#### LPR: The "Cost Plus" Calculation

* The LPR is also determined by a panel (similar to the SHIBOR panel, usually 18-20 banks), but the inputs are different.
* The Input (The Formula): Each bank submits a quote based on this logic:

$$
\text{Quote}=\text{Bank's Fund Cost} + \text{Risk Premium}
$$

where "risk premium" is often referred as spread.

Bank's fund cost is determined by

* Old Logic: Cost of Funds was strictly the MLF Rate.
* New Logic: Cost of Funds is a blend of the 7-Day Repo and Deposit Rates.

<div style="display: flex; justify-content: center;">
      <img src="imgs/cfets_lpr_20260129.png" width="30%" height="60%" alt="cfets_lpr_20260129" />
</div>
</br>
