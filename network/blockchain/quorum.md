# JP Morgan’s Quorum

In 2016, J.P. Morgan open-sourced a private, permissioned blockchain protocol called Quorum, derived from the Ethereum blockchain with a focus on institutional-grade performance, privacy, and security requirements. 

## Business Motivations and Benefits

Kinexys is a blockchain-based business unit focused on building and commercializing blockchain-based products, solutions, and infrastructure for J.P. Morgan’s clients. It consists of three in-production blockchain-based networks that provide solutions in the areas of payments, data sharing, and multi-asset settlements:

* Kinexys Link, a network for payment-related data sharing between banks
* Kinexys Digital Payments, a blockchain-based payment rail for institutional clients
* Kinexys Digital Assets, an asset tokenization platform that has three applications: Digital Financing (settlement of repo (repurchase agreement) transactions), Tokenized Collateral Network (tokenizing assets to use as collateral), and Digital Debt Service (issuance, settlement, and life-cycle management of debt instruments)
* Kinexys Labs, an incubator for new blockchain-based solutions

Quorum is the customized blockchain forked from Ethereum.
Quorum can run *solidity* smart contracts.

The Digital Financing (settlement of repo transactions) business currently processes approximately USD 2 billion on average in transaction volume on a daily basis in its repo markets and has surpassed USD 1.75 trillion in total volume since its launch in December 2020.

### Benefits in REPO (Repurchase Agreement) Transactions

REPO trade is characterized by a form of DVP (Delivery vs Payment) that one party lends securities and receives money, while the counterparty does the opposite (borrow securities and lend money).
For the trade quantity is often large, e.g., millions of dollar, usually the settlement date is T+2 that when a trade is executed, it takes 2 days for both parties to exchange money vs securities/goods.

The use of tokenization enables a unique combination of execution and settlement in the same venue and more broadly allows for further programmability.

According to the firm’s own assessment, one of its participants using its Digital
Financing platform experienced a reduction in its intraday financing costs of
50%–60% compared with its traditional intraday credit funding solution.

It has four major benefits.

* No reconciliations between transaction parties
* Reduced risks from collateral inventory mismanagement
* Increased trade lifecycle transparency
* Potential profits from precise timing on collateral trading

### Tokenization Process and Trade Life Cycle

1. Security Segregation/Transfer
    * The triparty custodian moves the assets from the regular account of the broker/dealer to an account of the collateral token agent (A separate custodian serving as the subcustodian is linked to the Kinexys platform)
2. Money Segregation/Transfer
    * The party must fund its blockchain deposit account (BDA) via Kinexys Digital Payments through a transfer of cash from its demand deposit account (DDA). 
3. REPO First Leg Trade Execution 
    * The lender and buyer negotiate terms and cryptographically sign the terms of the trade, which includes encoding into the smart contact the agreed-upon settlement and maturity timing.
    * The cash and entitlement to ownership of the collateral are **atomically settled** at the designated settlement time.
4. Between REPO Legs
    * The cash borrower withdraws its cash from the blockchain deposit account
    * The lender maintains entitlement to the collateral in tokenized form
5. REPO Second Leg Trade Execution 
    * In anticipation of maturity, the platform initiates a transfer of the cash plus interest from the cash borrower’s demand deposit account to the blockchain deposit account, followed by the smart contract triggering the maturity of the trade.
    * Digital Financing exchanges the principal plus interest for the collateral.
    * The involved parties can choose to remove their assets or leave them on the platform.

## The Tech of JP Morgan’s Quorum

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

#### Block Proposal in Quorum

Because Quorum is a permissioned enterprise blockchain, its block proposal process differs from public Ethereum (which uses Proof of Stake) and depends entirely on the consensus mechanism configured for the network (usually IBFT/QBFT or Raft).

* In IBFT / QBFT (Byzantine Fault Tolerance)

The network rotates the "Proposer" role among the approved Validator nodes in a round-robin fashion.
Once 2/3 of validators agree, the block is finalized.

By default, QBFT may produce empty blocks (heartbeats) to ensure the network is alive, even if there are no transactions.

* In Raft (Crash Fault Tolerance)

There is a single Leader (Minter) node elected by the cluster. Only the Leader can propose blocks.

## The Genesis

`genesis.json` file is the blueprint of a new blockchain. It tells every node exactly how the network starts (Block 0) and what rules to follow.

Below is an example of Istanbul standard `genesis.json` with enhanced privacy management that features:

* Uses QBFT Consensus.
* Has 5-second block times.
* Supports Private Transactions (Tessera).
* Allows Large Smart Contracts (64KB).
* Has specific accounts pre-loaded with money for testing.

```json
{
    "config": {
      "chainId": 1337,
      "homesteadBlock": 0,
      "eip150Block": 0,
      "eip150Hash": "0x0000000000000000000000000000000000000000000000000000000000000000",
      "eip155Block": 0,
      "eip158Block": 0,
      "byzantiumBlock": 0,
      "constantinopleBlock": 0,
      "istanbul": {
        "epoch": 30000,
        "policy": 0,
        "ceil2Nby3Block": 0
      },
      "txnSizeLimit": 64,
      "maxCodeSizeConfig" : [
        {
          "block" : 0,
          "size" : 64
        }
      ],
      "isQuorum": true,
      "privacyEnhancementsBlock": 0
    },
    "nonce": "0x0",
    "timestamp": "0x5f1663fc",
    "extraData": "0x...",
    "gasLimit" : "0xf7b760",
    "difficulty": "0x1",
    "mixHash": "0x63746963616c2062797a616e74696e65206661756c7420746f6c6572616e6365",
    "coinbase": "0x0000000000000000000000000000000000000000",
    "alloc": {
      "fe3b557e8fb62b89f4916b721be55ceb828dbd73" : {
        "privateKey" : "8f2a55949038a9610f50fb23b5883af3b4ecb3c3bb792cbcefbd1542c692be63",
        "comment" : "private key and this comment are ignored.  In a real chain, the private key should NOT be stored",
        "balance" : "0x130EE8E7179044400000"
      },
      ...
      "0xf0e2db6c8dc6c681bb5d6ad121a107f300e9b2b5": {
        "balance": "1000000000000000000000000000"
      },
      ...
    },
    "number": "0x0",
    "gasUsed": "0x0",
    "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000"
  }
```

where

* `chainId: 1337`: This is the unique ID for your network. 1337 is the standard default for private development networks (famously used by Geth/Ganache).
* `homesteadBlock` ... `londonBlock`: These settings define which Ethereum Hard Forks (EIPs) are active.
* `qbft`: This defines the Consensus Mechanism.
    * `blockPeriodSeconds: 5`: A new block will be minted every 5 seconds.
    * `epochLength: 30000`: Every 30,000 blocks, the validators reset votes and "checkpoint" the chain state.
* `isQuorum`: true: Explicitly marks this as a GoQuorum network (as opposed to standard Geth).
* `transitions`: These are Quorum-specific overrides.
    * `transactionSizeLimit` / `contractSizeLimit: 64`: This sets the max size to 64KB. This is double/triple the standard Ethereum limit (usually 24KB or 32KB). Private enterprise networks often need to deploy much larger smart contracts.
* `privacyEnhancementsEnabled: true`: Tells the node to look for the Tessera Private Transaction Manager. If it is `false`, the quorum node on this genesis cannot do private transaction.

### About `extraData`

About `"extraData": "0x..."`, for QBFT, the `extraData` field is an RLP List containing exactly 5 items:

```txt
RLP([ Vanity, Validators, Vote, Round, Seals ])
```

A very useful case is that in `extraData` field what validator node addrs are permitted.
If not listed in `extraData`, a quorum node does not join in validation (a non-validation quorum node can be used as RPC node to do read-only operation).

### About `alloc`

This section pre-funds specific accounts with ETH (gas tokens) at Block 0.
For example, `"0xf0e2db6c8dc6c681bb5d6ad121a107f300e9b2b5"` addr is granted a large sum of init ETH.

### Other Header Fields

* `mixHash: 0x6374...`: decodes to string: "critical byzantine fault tolerance"
* `difficulty: 0x1`: Since there is no Proof-of-Work mining, the difficulty is set to the lowest possible value so blocks can be validated instantly.
* `gasLimit: 0xf7b760`: (Decimal: ~16,234,336). The maximum amount of gas allowed in a single block.
* `coinbase: 0x000...`: The address that receives mining rewards. In many enterprise Quorum setups, gas is free (gasPrice = 0), so this is often left as the zero address.

## Transaction Flow Explanation by `npx quorum-dev-quickstart` Example

In the example of `npx quorum-dev-quickstart`, there are 8 nodes:

* 1 RPC node: a quorum as the front to receive RPC requests
* 3 Member nodes: quorum + tessera nodes (Member Node A, B, C)
* 4 Validator nodes: quorum validation nodes (Validator Node D, E, F, G)

All the 8 nodes sync and share the same blockchain.

### RPC Node Transaction Flow in Quorum Network

The RPC node is just a regular Quorum process.
The separation of the RPC node from the other four validator nodes is out of the safety and performance concerns to isolate Validation from heavy blockchain read operation.

#### Inbound Synchronization (Receiving Blocks)

1. A Validator (e.g., Validator 1) creates a block and then broadcast it to its peers.
2. The RPC node receives the block, verifies the cryptographic signatures (checking that >= 2/3 of validators signed it), and executes the public transactions inside to update its local copy of the Public State.

P.S., The RPC always lags slightly behind the validators (by milliseconds) but node maintains a full, valid copy of the public blockchain.

#### Outbound Data Sharing (Broadcasting Transactions)

1. The Mempool (Transaction Pool): The RPC node accepts user transaction request and places it in its local mempool.
2. The RPC node immediately "gossips" (broadcasts) this pending transaction to all connected peers, specifically the 4 Validators.
3. Same as in inbound synchronization that having done validation, the validators sync the new block back to the RPC node

### Private Transaction Flow in Quorum Network

The above RPC node transaction flow does NOT cover private transaction with quorum + tessera.

If user queries a Public variable on the RPC node, it returns the correct data.
If user queries a Private variable (managed by a Member node), the RPC node cannot see it. It only sees the "hash" of the private transaction in the block but has no way to decrypt the data.

In biz-semantics, the wallet addr concept is replaced with member node addr.
This is for that in private enterprise blockchain network, each company will deploy its owned member nodes, and biz unit is arranged by node not by wallet.

#### User Transaction Initiation

Assume User/company A (Node A) needs to do a private transaction with user/company B (Node B).

User/company sends a transaction contained `privateFor` parameter containing Node B's public key (Tessera key).
Quorum A detects `privateFor`. It knows this is not a standard Ethereum transaction. instead of broadcasting it immediately, it passes the transaction payload to its local Tessera A.
Neither RPC node nor validator node has tessera, hence this private transaction to RPC node will fail.

Node A needs to send the transaction to Tessera-paired nodes (Member Node A, B, C), not the RPC node nor validator node.

#### Privacy Manager Processing (Off-Chain)

1. Key Generation: Tessera A generates a random Symmetric Key for this specific transaction.
2. Encryption: It encrypts the transaction payload (the smart contract data) using this Symmetric Key.
3. Key Wrapping:
    * It encrypts the Symmetric Key using Node A's public key.
    * It encrypts the Symmetric Key using Node B's public key.
4. Distribution: Tessera A talks directly to Tessera B (via HTTPS/GRPC) and sends the encrypted payload + B's encrypted key.
    * Note: Tessera C is never contacted.
5. Hashing: Tessera A calculates the SHA3-512 Hash of the encrypted payload. This hash acts as the unique ID for the data.
6. Return: Tessera A returns this Hash to Quorum A.

#### Blockchain Propagation (On-Chain)

The private part of this transaction has done transmission, and the content is hashed. 
Member Node A now broadcasts the "Hash Transaction" to the network (same as a typical transaction except that the payload is just a hash).

1. The Validators receive the transaction via gossip. They see it comes from Member A and contains a data payload (the Hash).
2. The current Block Proposer (one of the 4 validators) includes this transaction in a new block.
3. The "Blind" Validation: The Validators check the signature and nonce. They do not check the private payload because they only see a Hash. 

## Quorum Implementation: Tokenized Collateral Network (TCN)

* Digital Twins: Lock the real asset in a "Special Purpose Vehicle" (SPV) or custody account and mint a "Digital Twin" (token) on the blockchain.
* Atomic Settlement: The exchange of collateral (Token A) for cash/asset (Token B) happens in a single transaction. If one fails, both fail. This removes counterparty risk.
* DvP (Delivery vs. Payment): Trade about asset delivery in exchange for cash payment

The smart contract is designed in such as way that multiple assets and currency cash are exchanged in an atomic trade.

Execution of `executeAtomicTrade` is atomic that it must either succeed or fail at all, and there is no "half-finish" that some assets are sent out but no matched cash received.
In this example below, if `_settleSellerLeg` is first to run and is successful, but the second leg `_settleBuyerLeg` fails, `_settleSellerLeg` has no effect.
Only the whole `executeAtomicTrade` finishes so that data is persisted.

```solidity
contract DvPSettlement {
    
    function executeAtomicTrade(
        address seller,
        address buyer,
        address[] calldata sellerTokens, // Bonds
        uint256[] calldata sellerAmounts,
        address[] calldata buyerTokens,  // Cash
        uint256[] calldata buyerAmounts,
        uint256 nonce
    ) external nonReentrant {
        _validateLegs(seller, buyer, sellerTokens, sellerAmounts, buyerTokens, buyerAmounts);

        bytes32 tradeHash = _computeTradeHash(seller, buyer, sellerTokens, sellerAmounts, buyerTokens, buyerAmounts, nonce);
        TradeApproval storage approval = _tradeApprovals[tradeHash];
        require(approval.sellerApproved && approval.buyerApproved, "DvP: missing approvals");
        require(!approval.executed, "DvP: already executed");
        approval.executed = true;

        _settleSellerLeg(seller, buyer, sellerTokens, sellerAmounts);
        _settleBuyerLeg(buyer, seller, buyerTokens, buyerAmounts);

        emit TradeExecuted(seller, buyer, nonce, block.timestamp);
        delete _tradeApprovals[tradeHash];
    }
}
```


