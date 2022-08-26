# Repo risk

Credit risk is the possibility of a loss resulting from a borrower's failure to repay a loan or meet contractual obligations.

Repo risk mainly associates with liquidity and price fluctuation of collaterals, that the counter party might default when his collateral's value fall. A margin call is invoked: if the security loses value, the lender can demand more money (or securities) to protect its investment.

Repo price is usually computed at the base of collateral, adding repo rate/interest (as benefit) and haircut (as risk aversion compensation), as well as frictional costs (a broad horizon considerations of costs) such as commissions, consultation/research fees, etc.

In real world markets, there are third parties (a.k.a *Tri party*) to liaison with between repo lenders and borrowers, managing non-gov bonds (e.g. corporate bonds) as collaterals. For gov bonds, banks and institutions can directly transact with governments, hence no need of engaging other parties.

## Standard Repurchase Agreement (SRA)

Two banks $b_1$ and $b_2$ agree congruently about 
1. repo rate $r$
2. haircut $h$
3. collateral assets $y$
4. frictional costs $fr$

by signing a contract named *Standard Repurchase Agreement* (SRA,）denoted as $C(r,h,y, fr)$.

Consider a money market over three dates $d_0$, $d_1$, $d_2$, together with $A_{1+m}$ assets: cash and $m \ge 1$ collaterals.

Cash is riskless carrying no interest. Collaterals are prone to risks and being illiquid.

On $d_2$, a principle $p$ plus money for interest rate $r$ should be paid to claim collateral assets.

### Defaults 

Define a state $w$

* $w = G$ means a good state, neither the lender nor the borrower defaults
* $w = B$ means borrower default, that borrower does no have enough money to repay the principle plus an interest
* $w = L$ means lender default

In repo, one bank $b_1$ borrows (denoted as $b_B$) from a lender bank $b_2 = b_L$. The state of defaults can be expressed $\pi_w=\pi_w(b_B, b_L)$ which describes the default probability at $d_2$, where $w \in \{G,B,L\}$ and clearly $\pi_G+\pi_B+\pi_L=1$.

## Default Scenarios 

The below scenarios should take into consideration for drafting a SRA.

### Scenario: Two-sided credit risk

Two-sided credit risk ($\pi_B > 0, \pi_L > 0$) describes both borrower and lender banks have default probability. Both lender and borrower are exposed to certain risks.

### Scenario: Netting

Define $v_a$ as the replacement cost of a collateral portfolio at $d_2$, conditional on the lender's default; $v_b$ as the liquidation value at $d_2$ of a collateral portfolio, conditional on the borrower's default.

Given the collateral's liquidation value at $d_2$, 
* $w=B$, lender suffers a potential loss if $v_b < p+r$
* $w=L$, borrower suffers a potential loss if $v_a > p+r$

### Scenario: Subordination

Windfall profits are a sudden and unexpected spike in earnings, often caused by a one-time event that is out of the norm. 

Either $w=B$ or $w=L$, counterparty can earn windfall profits for increases/decreases in collaterals' values.

### Scenario: No Windfall Profit

Here define two utility first-order differentiable functions $u_B(.)$ and $u_L(.)$ for borrower and lender, respectively; $\tilde{u}_B$ and $\tilde{u}_L$ are the lender's and borrower's uncertain terminal utility at the time of contracting.

Given an interest rate $r$, there are utility expectations for the borrower and lender:
$$
\begin{align*}
E(\tilde{u}_B) &= \pi_G u_B(-r)+ \pi_L E\big(\tilde{u}_B(min\{1-v_a;-r\})\big)
\\
E(\tilde{u}_L) &= \pi_G u_L(r)+ \pi_B E\big(\tilde{u}_L(min\{v_b-1;r\})\big)
\end{align*}
$$

The above probability model states that the expected utility should account for good state interest plus bad states for lender or borrower, respectively.

### Scenario: Liquidity ranking

There exists $m$ assets together as collaterals defined in a SRA. A ranking is assigned to the $m$ assets to describe the liquidity and riskiness of these assets.

Good collaterals are first used up.

## Practical risk analysis

Compare similar repo products on the market, and monitor the fluctuations of prices and volumes.
