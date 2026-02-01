# Asset Tokenization

## Business Motivations

In the financial world for two banks to settle a trade, there exist below pain points.

### The "Single Source of Truth" vs. "Reconciliation Hell":

In the traditional banking world (SWIFT/Correspondent Banking), money does not actually "move."

Traditional Way:

1. Bank A debits its internal database (Leger A): "-10M".
2. Bank A sends a secure message (SWIFT) to Bank B.
3. Bank B receives the message and credits its internal database (Ledger B): "+10M".
4. The Problem: What if Bank A’s database says sent, but Bank B’s system crashed? Or what if there is a typo? Or a dispute?

Result: Banks employ thousands of operations staff just to do Reconciliation—checking if Spreadsheet A matches Spreadsheet B at the end of every day.

Blockchain Way:

1. There is no "Database A" and "Database B." There is only The Ledger.
2. When the transaction is confirmed on Quorum, both banks look at the exact same record. They share a "Golden Source" of truth.

Benefit: Reconciliation is instant and automatic; the shared ledger is the proof it arrived.

### The "Double Spend" Problem (Without a Central Bank)

If Bank A sends a digital file saying "Here is $10M" to Bank B via an API, what prevents Bank A from sending that same file to Bank C five minutes later?

Traditional Way:

* There needs a trusted third party (an Admin), like a Central Bank or a Clearing House (e.g., The Fed, CHIPS, Euroclear). Both banks trust this Admin to track who owns what. This adds fees and delays (T+2 settlement days).

* Blockchain Way:

The Network (Bank C and D) acts as the notary.

### Delivery vs. Payment (DvP) – The "Atomic Swap"

Banks rarely just "send money" for no reason; they usually send money to buy something (like a Bond or Stock), i.e., to do DvP trade.

Traditional Way:

1. Bank A sends Cash to Bank B. (Wait 1 day).
2. Bank B sends the Bond to Bank A.
* Risk: What if Bank B goes bankrupt in that 1-day window? Bank A lost their cash and didn't get the bond. This is called Herstatt Risk (Settlement Risk).

Blockchain Way (Smart Contracts):

* Write smart contract: "Move Cash from A to B IF AND ONLY IF Bond moves from B to A."
* This happens in the exact same millisecond (Atomic Settlement).
* Benefit: Counterparty risk is effectively eliminated,  that this counterparty cannot just withdraw the trade for settlement happens in milliseconds.

#### The Traditional DvP: Entities Involved in Clearing and Settlement

* Asset manager acts as a fiduciary to instruct and execute trades on behalf of client accounts they have authority over.
* A trading venue is a regulated and authorized facility or platform that allows for securities to be bought and sold among various parties (e.g., a stock exchange).
* A clearinghouse is responsible for finalizing trades, settling trade accounts, and reporting trade data.
* Custodian banks and depositories, including central securities depositories, generally hold the assets (in physical or electronic form) and maintain a record of client ownership. 

## Real World Asset (RWA) Tokenization on Chain

Asset tokenization definition:

> The process of recording claims on real or financial assets that exist on
> a traditional ledger onto a programmable platform

The value of token:

* Tangible RWA asset (off-chain)

> The token is a mirror of the value and ownership rights that are associated with the asset and acts
> as a link between the blockchain and the real world, allowing the token to be
> tradable and transferable within the blockchain ecosystem.

* Native to the blockchain (on-chain)

> Native tokens that are used as a means of exchange, as a store of value, to execute contracts, or to
> participate in the governance of the protocol.

### Real World Asset (RWA) Tokenization Process

1. Origination, involves pricing and auditing the underlying asset and structuring the deal, including determining fee structures, capital commitment requirements, understanding relevant regulations, and tax implications, all of which can be encoded on-chain.

2. Selecting the token issuance service provider, the KYC/AML8 vendor, the custodian or private trust company, and the secondary market provider.

3. asset can be tokenized, including choosing
a token standard,9 choosing a public or private network, writing the
smart contract10 according to the specifications of the deal structure, and
satisfying compliance and regulatory requirements.

4. Finally, the tokens can be issued directly to investors or listed on a market.
Investors must set up their wallets to receive the tokens, with the underlying
asset remaining with the custodian.

## Some Terminologies of Distributed Ledger

* Permissionless Any node can download the ledger and validate transactions.
* Permissioned Permission is required to download the ledger and validate transactions.
* Public Any node can read and initiate transactions on the ledger.
* Private Only a selected group of nodes can read and initiate transactions.
* Nonhierarchical Each node has a full copy of the ledger.
* Hierarchical Only designated nodes have a full copy of the ledger.
* Open source Anyone can suggest changes to the code underpinning the ledger platform.
* Closed source Only trusted entities can see and make improvements to the code underpinning the ledger platform.

## Regulation and Compliance (by 2025)

Reference:

https://rpc.cfainstitute.org/sites/default/files/docs/research-reports/bandi_tokenization_partii_online.pdf

| Jurisdiction | Summary of Policy | Estimated Attitude |
| :--- | :--- | :--- |
| **Switzerland** | **Integration into Existing Law:** Rather than creating new laws, Switzerland updated 10 existing federal laws (the DLT Bill) to accommodate blockchain (e.g., Code of Obligations). It introduced "DLT trading facilities" under FINMA supervision to allow trading and custody without needing additional licenses. | **Highly Supportive & Integrated**<br>Viewed as a "top location" and "key financial center." The approach prioritizes legal certainty and business enablement, fostering the "Crypto Valley" ecosystem. |
| **European Union** | **Comprehensive Standardization (MiCA):** The EU enforced the Markets in Crypto-Assets Regulation (MiCA) in 2023. It focuses on the "same activities, same risks, same rules" principle. It mandates Legal Entity Identifiers (LEI) and strict transparency, excluding assets already covered by existing financial instrument laws. | **Regulated & Bureaucratic**<br>Proactive in establishing a standardized, extensive rulebook. The attitude is one of strict oversight and uniformity to ensure transparency and compliance across member states. |
| **United Kingdom** | **Holistic Safety & Consumer Protection:** The UK focuses on stablecoin regulation, AML compliance (Travel Rule), and strict controls on financial promotions (risk warnings, cooling-off periods). The Law Commission is working to define digital assets as personal property. | **Cautious & Protective**<br>While supportive of innovation, the UK prioritizes market integrity and consumer safety, acting swiftly against non-compliant promotions and emphasizing risk management. |
| **India** | **Evolving & Reassessing:** After an overturned banking ban, India integrated Virtual Digital Assets (VDAs) into AML regulations in 2023. While the Reserve Bank remains concerned about macroeconomic risks, the government is currently revisiting its stance to align with global trends. Special zones (GIFT City) are exploring tokenization. | **Ambivalent / Hesitant**<br>The attitude is shifting from hostility toward a "flexible and adaptive" approach, though it remains cautious regarding financial stability and private currencies. |
| **Mainland China** | **Dual Approach (Ban vs. Property Rights):** The government maintains a strict ban on cryptocurrency trading and transactions, labeling them illegal and a threat to financial stability. However, the judicial system legally recognizes virtual assets as "property" protected by law. | **Restrictive / Prohibitive**<br>Hostile toward the *activity* (trading/speculation) to maintain financial control, but respectful of the *asset* (individual ownership rights) within the legal system. |
| **Singapore** | **Pragmatic & Judicial:** The framework relies on the Payment Services Act (PSA) and court rulings that classify digital assets as property ("things in action"). Regulations focus heavily on custody, segregating customer assets, and trust accounts. | **Pragmatic & Secure**<br>The stance is pro-business but deeply focused on safety and resilience. It aims to foster a "secure" environment through practical, high-standard regulation rather than sweeping new legislation. |
| **United Arab Emirates (UAE)** | **Fragmented Ambition:** Policies vary by jurisdiction. Federal bodies (SCA/CBUAE) have strict licensing; Dubai (VARA) has a dedicated virtual asset regulator; Abu Dhabi (ADGM) is proactive with stablecoin rules. There is friction between federal bans on using crypto for payments and free-zone innovation. | **Ambitious but Complex**<br>The UAE wants to be a global leader and innovation hub, but the text notes that its complex, multi-layered jurisdictional approach "may not be conducive" to doing so efficiently. |
| **Hong Kong SAR** | **Proactive "Crypto-Readiness":** Recognized as the most "crypto-ready jurisdiction" in 2023. It utilizes dual licensing regimes for security and non-security tokens, recognizes crypto as property, and is actively developing a stablecoin issuer regime. | **Very Positive / Aggressive**<br>The government has an explicit policy to develop the sector, showing a willingness to adapt property laws and licensing to attract business and ensure clarity. |
| **United States** | **Pivoting to Leadership (2025):** Following a 2025 Executive Order, the US is shifting from enforcement-heavy regulation to a "technology-neutral" framework. The SEC is establishing a task force to create clear guidelines, and Congress is debating the GENIUS Act to regulate stablecoins. CBDCs are explicitly banned. | **Hegemonic / Reclaiming**<br>Historically fragmented and ambiguous, the attitude in 2025 has shifted to "Strengthening American Leadership." The focus is now on protecting the US dollar's dominance via stablecoins and fostering industry growth. |