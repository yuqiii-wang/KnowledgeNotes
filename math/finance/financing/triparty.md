# Custodian and Triparty

Prerequisite concepts:

* Custodian: A financial institution that holds customers' securities for safekeeping to prevent them from being stolen or lost. They track who owns what
* Tri-party Agent: An intermediary that administers the transaction between two principals (Buyer and Seller).
* Delivery Versus Payment (DVP): Delivery of securities/commodities vs money payment happens as simultaneously as possible to eliminate principle risk that, for example, buyer has paid money to seller already but seller soon goes bankrupt.

## Triparty REPO Transaction Flow

Reference: https://bk.bnymellon.com/rs/353-HRB-792/images/BNY_Triparty_Repo_Brochure.pdf

<div style="display: flex; justify-content: center;">
      <img src="imgs/triparty_repo_trans_flow.png" width="50%" height="40%" alt="triparty_repo_trans_flow" />
</div>
</br>

Below is an example of BNY (Bank of New York Mellon) as a triparty facilitating a REPO transaction

1. The seller enters into a repo trade with a cash provider (buyer), agreeing to post securities as collateral in exchange for cash funding.
2. In our capacity as triparty agent, both the seller and the cash provider supply us with the details of the repo trade.
3. Triparty (BNY) matches the trade instructions to verify all trade terms are consistent.
4. The cash provider transfers cash to BNY Mellon.
5. The seller delivers the collateral that will support the repo trade to us.
6. Triparty (BNY) places the pledged collateral in a segregated account.
7. Triparty (BNY) confirms to the cash provider that the securities supporting the repo have been placed into the segregated account.
8. The cash is transferred to the seller. 

### Collateral Selection Algo: "Conditioned Cheapest"

Given a pool of collateral securities, triparty first checks a num of requirements, then pick up the "cheapest" few.

#### Condition Filter

The engine takes the Eligibility Schedule (the contract agreed upon by Buyer and Seller) and applies it as a filter over the dealer's inventory.

For example, both buyer and seller agree collateral on meeting these criteria shall be traded, and triparty will set these conditions as first priority.

* Rule 1: Must be USD.
* Rule 2: Must be rated A- or better.
* Rule 3: No single issuer can exceed 10% of the basket (Concentration Limit).
* Rule 4: No bonds maturing in > 30 years.

#### The Optimization (Cheapest-to-Deliver)

Having satisfied the bi-party (both buyer and seller) agreed conditions, triparty selects the "cheapest" collateral to dealer/buyer.

Example: If the Buyer accepts both Corporate Bonds and Treasuries, the engine will automatically grab the Corporate Bonds from the dealer's box to satisfy the $50M. It only uses the Treasuries if the Dealer runs out of Corporate Bonds.

### The DVP Settlement Moment

The DVP is the execution phase.

The freeze:

* Buyer's money and seller's bonds are frozen 

The Swap:

* Cash Leg: A debit is posted to the Buyer's Demand Deposit Account (DDA); a credit is posted to the Seller's DDA.
* Securities Leg: The specific bonds selected in Step 4 are moved from the Seller's "Main Account" to a Segregated Collateral Account linked to the Buyer.
