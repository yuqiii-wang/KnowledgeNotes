# JP Morgan’s Quorum

Quorum is forked from Ethereum focusing on enterprise private blockchain network.
It has enhancements on transaction privacy.

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

## The Solution: JP Morgan’s Quorum

JP Morgan took the public Ethereum code (Geth) and modified it to fit private blockchain network needs: *Quorum*

Consider a number of large financial institutions (Bank A, B, C, D) together founded a private blockchain network to facilitate fund transfer, the flow goes as below

Assume Bank A needs to transfer fund to Bank B:

1. Before any money moves, a Private Smart Contract (let's call it TokenContract) must exist between Bank A and Bank B; Bank C and Bank D cannot see this contract
2. Transaction Init: Bank A initiates a transfer
    1. Bank A's Application constructs a standard Ethereum transaction payload: `function transfer(address _to, uint256 _value)`.
    2. It wraps this payload in a Quorum request, adding the privacy flag: `privateFor: ["Bank B's Tessera Key"]`.
    3. This request is sent to Bank A's Quorum Node via RPC.
3. Quorum Work: Bank A's Quorum Node sees the `privateFor` field and hands the data to its Tessera (Privacy Manager) node.
    1. encryption key generation and assignment to Bank A and Bank B
    2. Bank A can communicate to Bank B directly in encryption of fund transfer detail.
    3. Once Bank B acknowledges receipt, Bank A's Tessera returns a SHA3-512 Hash of the encrypted data to the Quorum Node.
4. Bank A's Quorum Node now creates the actual blockchain transaction.
    1. ONLY Bank A and Bank B know the transaction detail
    2. Bank C and Bank D only receive a hash record of this transaction and public-releasing data

In the above flow, Bank A acts as the signer of the trade, and Bank C and Bank D acts the notary.

## The Architectural Components of Quorum

### Node Types

* Members: These are the legal entities or companies (e.g., Bank A, Supply Chain B). They own the nodes but don't necessarily perform the technical work of creating blocks.
* Validators: These are the specific nodes authorized to propose and seal blocks. They take transactions, order them, and write them into the ledger.

A member has both GoQuorum and Tessera components, while a validator only has the GoQuorum component.

### Inside Node Communication

The Quorum Node (GoQuorum)

* Based on: A fork of Geth (Go-Ethereum).
* Role: It maintains the blockchain, validates blocks, and executes transactions; besides, has a privacy control function
* Uses LevelDB (standard Ethereum key-value store). However, unlike Ethereum, it maintains two separate Merkle Patricia Tries:
    * Public State Trie: Synchronized globally across all nodes.
    * Private State Trie: Local to the node; contains state data only for the private contracts this node is party to.

The Privacy Manager (Tessera)

* Role: A separate Java-based service that runs off-chain. It handles the heavy lifting of encryption, decryption, and peer-to-peer distribution of private data.
* Transaction Manager: Manages the storage and communication of encrypted payloads.
* Database (Private Payloads): Uses a SQL Database (via JDBC), e.g., PostgreSQL, Oracle, or MySQL.

### Common Questions

#### Why need to mint a coin before executing a smart contract

Quorum is a "fork" of Ethereum. This means it uses the Ethereum Virtual Machine (EVM) to execute smart contracts.

In the original Ethereum design, the "coin" (Ether) serves two mandatory technical purposes:

* The Halting Problem (Gas): To prevent someone from accidentally or maliciously writing an infinite loop that crashes the whole network, every operation costs "Gas." In public Ethereum, you buy Gas with Ether.
* Account Balances: The EVM code is hard-coded to check if an account has enough "balance" to pay for the computation it is requesting.

To make the code run without rewriting the entire Ethereum core, developers often "mint" a massive amount of "fake" coins to the participating nodes so the "Gas" check always passes.
Quorum handles this by allowing a Gas Price of 0.

#### The Flow of Smart Contract Deployment

1. The network starts. The `genesis.json` file pre-allocates (mints) a massive amount of "fake" Ether to user wallet, so that there is no need of worrying about transaction fee/gas.
2. Smart Contract Deployment

#### The Flow of Smart Contract Execution


## Quorum Implementation: Tokenized Collateral Network (TCN)

* Digital Twins: Lock the real asset in a "Special Purpose Vehicle" (SPV) or custody account and mint a "Digital Twin" (token) on the blockchain.
* Atomic Settlement: The exchange of collateral (Token A) for cash/asset (Token B) happens in a single transaction. If one fails, both fail. This removes counterparty risk.
* DvP (Delivery vs. Payment): Trade about asset delivery in exchange for cash payment

```solidity

```
