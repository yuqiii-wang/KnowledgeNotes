# Ethereum

*Ethereum* is a decentralized, open-source blockchain with smart contract functionality.

## Ethereum Virtual Machine (EVM)

The Ethereum Virtual Machine or EVM is the runtime environment for smart contracts in Ethereum.

### Physical Storage

The EVM does not use a SQL database (like Postgres). It uses high-performance, embedded Key-Value (KV) Stores. The specific technology depends on running "Client" (node software).

|Client|Database Engine|Architecture Type|Why?|
|:---|:---|:---|:---|
|Geth (Go-Ethereum)|LevelDB|LSM Tree (Log-Structured Merge-Tree)|Created by Google. Optimized for high write throughput (appending data).|
|Nethermind / Besu|RocksDB|LSM Tree|Created by Facebook. A fork of LevelDB optimized for multi-core CPUs and fast NVMe SSDs.|
|Erigon / Reth|MDBX|B+ Tree|A super-fast version of LMDB. It uses Memory Mapping, allowing the database to be read directly from RAM/Disk without copying buffers.|

### The EVM Execution Components

When a transaction runs, the EVM instantiates a new Execution Context.

There is a Program Counter (PC) pointing to the next Opcode in the Bytecode.
And Ethereum chain charged transaction fee/gas is based on/proportional to how many bytecode instructions it has executed

For example, given a specific line of Solidity code: `count = 5;`.
EVM compiles it into these bytecode:

```asm
PUSH1 0x05      // Push "5" (Value)
PUSH1 0x00      // Push "0" (The storage slot index for 'count')
SSTORE          // Save Value "5" to Slot "0"
```

Before executing the bytecode, EVM checks the wallet address does have enough money/gas as well as other sanity check.
Then, execution starts, and on reaching `SSTORE`, EVM marks it dirty in RAM, and finally commits the change on disk.

Popular engines are

|Client Software|Language|EVM Execution Engine|Physical Database|Storage Strategy|Key Strength|
|:---|:---|:---|:---|:---|:---|
|Geth (Go-Ethereum)|Go|JumpTable Interpreter (Standard Loop)|LevelDB (or Pebble)|MPT (Tree) (High Read Amplification)|Stability & Standards (The reference implementation)|
|Reth (Rust Ethereum)|Rust|revm (Modular & Extremely Fast)|MDBX|Flat KV (1 Disk Read per lookup)|Max Performance (Fastest sync & execution)|
|Nethermind|C# (.NET)|JIT Compiler (Just-In-Time optimization)|RocksDB|Paprika (Custom Optimized Trie)|Corporate / Windows (Great for institutional setups)|
|Erigon|Go|JumpTable (Forked from Geth)|MDBX|Flat KV (Separates Data from Trie)|Data Analytics (Efficient Archive Nodes)|

P.S., since v1.14, Geth separates execution and consensus components, while in v1.13, a lightweight consensus engine *Clique* in embedded.

### The EVM Consensus Components

The Consensus Client handles the Beacon Chain. It manages the validators, tracks the logic of who votes for what block.

|Client|Language|Developed By|Resource Usage|Key Strength|Best For|
|:---|:---|:---|:---|:---|:---|
|Prysm|Go|Prysmatic Labs (Offchain Labs)|Medium/High|User Experience. Great documentation, widely used.|Beginners & Geth Users|
|Lighthouse|Rust|Sigma Prime|Low|Performance. Extremely stable, safe, and fast.|Power Users & Stakers|
|Teku|Java|ConsenSys|High (RAM)|Enterprise. Built for institutional setups.|Banks & Institutions|

## Merkle Patricia Trie (MPT)

Merkle Patricia Trie (MPT) is the data structure used in Ethereum.
There is a world state MPT that stores wallet/contract addr -> block addr; in each block there are three trees to store actual data.

### Why Need MPT ? Isn't Hash Table with O(1) CRUD Better ?

* Need a hash to represent the whole DB at a moment to ensure data integrity

To calculate the "Hash of the Database," it needs to hash every single entry in the database together.
Even with change of one value (e.g., Alice pays Bob), it needs to re-read and re-hash the entire database (millions of accounts) to generate the new Root Hash.
It is $O(n)$.

With tree, there only needs to re-hash the nodes along the direct path from that leaf up to the root. The rest of the tree remains untouched. This makes calculating the new Root Hash $O(\log n)$.

For example, 

* Leaf nodes: A through H are the Accounts (Data).
* Level 1, 2, 3 are just Hashes (Summaries).

```txt
Level 3 (ROOT):             [      ROOT HASH       ]
                                   /      \
Level 2:                [HASH ABCD]        [HASH EFGH]
                         /      \            /      \
Level 1:            [HASH AB] [HASH CD]  [HASH EF] [HASH GH]
                     /    \     /    \     /    \     /    \
Level 0 (Data):     A      B   C      D   E      F   G      H
```

If a light client, e.g., a mobile phone app, needs to obtain proof of account A balance that has possible multiple transactions with account B, C, D, E, ..., the light client only needs `Hash( Hash( Hash(A + B) + HashCD ) + HashEFGH ) -> ROOT`, instead of downloading the whole database of leaf nodes.

If there is a change to account A, only `HashAB`, `HashABCD` and `HashRoot` need to update, other hashes do not need to update.

### Merkle Patricia Trie (MPT) Intro

Reference: https://ethereum.org/developers/docs/data-structures-and-encoding/patricia-merkle-trie/#prerequisites

* Merkle Proofs

In wallet use scenario, user need to check wallet balance and on new transaction updating the balance.

Remember that wallet balance is the aggregation result of all wallet transactions, not a single num.

Users need to verify transaction results without downloading the entire DB.
With Merkle Proof, this consists of the specific leaf (Alice's account) and the neighbor nodes along the path to the Root.

* Merkle tree

In cryptography and computer science, a hash tree or Merkle tree is a tree in which every "leaf" node is labelled with the cryptographic hash of a data block, and every node that is not a leaf (called a branch, inner node, or inode) is labelled with the cryptographic hash of the labels of its child nodes.

In short, nodes are the re-hashed of child nodes.

<div style="display: flex; justify-content: center;">
      <img src="imgs/merkle_tree.png" width="40%" height="30%" alt="merkle_tree" />
</div>
</br>

* Radix tree

In computer science, a radix tree (also radix trie or compact prefix tree or compressed trie) is a data structure that represents a space-optimized trie (prefix tree) in which each node that is the only child is merged with its parent.

In short, parent nodes contain prefix data of child nodes, complete data can be concatenated by traversing a path of a tree reaching to leaf nodes.

<div style="display: flex; justify-content: center;">
      <img src="imgs/radix_tree.png" width="30%" height="30%" alt="radix_tree" />
</div>
</br>

* "Merkle" Radix tree

A "Merkle" Radix tree is built by linking nodes using deterministically-generated cryptographic hash digests.
If the root hash of a given trie is publicly known, then anyone with access to the underlying leaf data can construct a proof that the trie includes a given value at a specific path by providing the hashes of each node joining a specific value to the tree root.

* Merkle Patricia Trie

Radix tries have one major limitation: they are inefficient.

To store one (path, value) binding where the path, like in Ethereum, is 64 characters long (the number of nibbles in bytes32), it will need over a kilobyte of extra space to store one level per character, and each lookup or delete will take the full 64 steps.

The Patricia trie introduced optimization addressing the inefficiency.

### Tries in Ethereum

All of the merkle tries in Ethereum's execution layer use a Merkle Patricia Trie.

#### World State Trie

There is one global state trie, and it is updated every time a client processes a block. In it, a path is always:

`keccak256(ethereumAddress)` and a value is always: `rlp(ethereumAccount)`, where RLP stands for *Recursive Length Prefix*. More specifically an Ethereum account is a 4 item array of `[nonce,balance,storageRoot,codeHash]`.

<div style="display: flex; justify-content: center;">
      <img src="imgs/world_state_account.png" width="40%" height="30%" alt="world_state_account" />
</div>

There are two types of accounts:

* Wallet Account/Addr: Used by actual persons 
* Contract Addr: Internally used smart contracts

<div style="display: flex; justify-content: center;">
      <img src="imgs/world_state_two_account_types.png" width="40%" height="30%" alt="world_state_two_account_types" />
</div>

#### Storage Trie

* Storage trie is where **all contract data** lives.
* There is a separate storage trie **for each account**.

Storage trie stores the smart contract execution results:

* EOAs (Externally Owned Accounts): Users with private keys (like your MetaMask wallet) have an empty Storage Trie. Their StorageRoot is the hash of an empty tree.
* Smart Contracts: Every time a user deploys a contract, a new, empty Storage Trie is created for it. As the contract executes and saves data, this specific tree grows.

#### Transactions Trie

There is a separate transactions trie for every block.
The key/value is `rlp(transactionIndex) -> transaction_data`.

For example, there are 50 transactions in a block.

* `rlp(1) -> transaction_data_1`
* `rlp(2) -> transaction_data_2`
* `rlp(3) -> transaction_data_3`
* ...
* `rlp(50) -> transaction_data_50`

These pairs are built into a Merkle Patricia Trie.

#### Receipts Trie

Every block has its own Receipts trie.

There is a 1-to-1 relationship between a Transaction and a Receipt.

## Ethereum Network Types

In the Ethereum ecosystem, the software (EVM) is identical across all versions, but the data (ledger) and the peers (network) are different.

|Feature|Mainnet (Production)|Testnets (Sepolia / Holesky)|Private Networks (Quorum)|
|:---|:---|:---|:---|
|Purpose|Real Economy & Business|Development & Debugging|Enterprise & Privacy|
|Currency|Real ETH ($)|Fake ETH (Worth $0)|No Value (or Custom Token)|
|Consensus|Proof of Stake (PoS)|PoS (Permissioned)|Proof of Authority (PoA)|
|Security|Global ($60B+ staked)|Low (Centralized validators)|High (Walled Garden)|
|Access|Public (Anyone)|Public (Anyone)|Restricted (Invite Only)|

## A Typical Ethereum Process

### A User Submitted Execution of A Transaction

In short, after a user submits a transaction to the blockchain:

1. The first blockchain computer on receiving it does preliminary validation, then broadcast to other nodes
2. A proposer blockchain computer on receiving the transaction, along with other user transactions, pick up the highest fee transaction and execute as many transactions as possible within a few secs
3. This proposer blockchain computer then broadcast the block of transaction to other blockchain computers for validation (re-execute all transactions to check hashes are matched)
4. If >66% of validation computers agree on the hash results, the block is justified; then finalized after next epoch ends

#### 1. Transaction Submission & The Execution Layer (EL)

The signed transaction is sent to an Execution Client/Node (e.g., Geth, Nethermind) via the JSON-RPC method `eth_sendRawTransaction`.

Then perform preliminary validation for wallet:

* Intrinsic Gas: Does gasLimit cover the data payload cost (21,000 base + 16 gas/non-zero byte)?
* Signature: Is the ECDSA signature valid and does it recover to the sender's address?
* Nonce: Does the transaction nonce match the sender's current on-chain nonce?
* Balance: Does the sender have enough ETH for value + gasLimit * maxFeePerGas?

Having done validation for wallet, the first execution node stores this transaction on its RAM named *mempool*, and broadcast (by  `devp2p` gossip protocol) to other nodes.
Each peer repeats the validation before adding it to their own local mempool.

Note: The mempool is purely an Execution Layer concept. The Consensus Layer is unaware of individual transactions at this stage.

#### 2. Validator Selection (The Consensus Layer)

Ethereum uses a slot-based time system. Time is divided into Slots (12 seconds) and Epochs (32 slots, ~6.4 minutes) in 2025.

* RANDAO: In every epoch, the Beacon Chain uses the RANDAO value (an accumulation of randomness contributions from validators) to deterministically shuffle the validator set.
* Committee Assignment:
      * Block Proposer: For each slot, exactly one validator is pseudo-randomly assigned as the Proposer.
      * Attesters: Other validators are assigned to Committees to vote on the validity of the block.
* Lookahead: Validators know their duties (proposing or attesting) at least one epoch in advance, allowing them to prepare their clients.

#### 3. Block Building

When the allocated slot arrives, the Proposer's Consensus Client (CL) (e.g., Prysm, Lighthouse) instructs its paired Execution Client (EL) to build a block.

The general rule is to select transactions from its `mempool` based on gas price (and MEV opportunities if using an external builder).
In 2025, Ethereum implements Builder-Proposer Separation (BPS) that builder computers bid to obtain the right to run building a block.

Consensus Layer (CL) typically wait for 4 secs to get EL execution results `ExecutionPayload`:

* `transactions`: List of RLP-encoded transaction byte arrays.
* `blockNumber`, gasUsed, stateRoot, receiptsRoot, logsBloom.
* `blockHash`: The Keccak-256 hash of the execution block header.

#### 4. Block Attestation and Finalization

The Proposer Consensus Client (CL) takes the `ExecutionPayload` and wraps it into a Beacon Block.

The Proposer signs this entire Beacon Block using their BLS signing key.

The signed Beacon Block is broadcast to the network via the `libp2p` gossip-sub protocol (Consensus Layer network):

* Verification: Other nodes (Attesters) receive the block. Their CL parses the Beacon Block, and their EL validates the `ExecutionPayload` (re-executing transactions to ensure the `stateRoot` matches).
* Attestation (Voting): If valid, Attesters broadcast an Attestation. This is a vote supporting the block as the new head of the chain.
* Aggregation: Instead of storing thousands of individual signatures, specialized validators called Aggregators combine identical attestations into a single Aggregate Attestation using BLS signature aggregation.

Justification and Finalization

* Justification: Once an epoch ends, if the pair of checkpoints (start and end of the epoch) receives >66% (super-majority) of the total active validator stake in attestations, the epoch is Justified.
* Finalization: If two consecutive epochs are Justified, the first one becomes Finalized.

### What Is Inside Block Building

By Builder-Proposer Separation (BPS) design, builder computer sort all transactions in their respective `mempool` and estimate the "effort".
A chosen proposer computer accepts The highest bid.

For example. given below two estimate, a chosen proposer computer accepts Builder Y's bid.

* Builder X constructs a block worth 1.0 ETH.
* Builder Y constructs a block worth 1.1 ETH.

There are relays/validators to make sure the bidder does NOT lie about its estimate, otherwise by PoS this lying builder computer will see penalty reducing its ETH stake to its wallet.

There is max gas limit (all-transaction gas fee indicates how computation-intensive it is to build to block) that this builder block will only finish smart contract execution.

#### Data: World State Trie and Storage Trie

* The World State Trie: Maps `Keccak256(Address)` -> `RLP(Account)`
    * Account: `[Nonce, Balance, StorageRoot, CodeHash]`
* The Storage Trie (For smart contract only): Maps `Keccak256(StorageSlot)` -> `RLP(Value)`
    * Every contract account has its own separate Storage Trie, pointed to by the `StorageRoot` field in the account.

#### Smart Contract Execution

When a transaction executes a smart contract that writes data (using opcode `SSTORE`), it does not immediately rewrite the Merkle Root. It modifies a "Dirty Map" in memory.

1. Fetch Account:
    * The EVM computes `addrHash = Keccak256(Tx.to)`.
    * It traverses the World State Trie to find the leaf node for `addrHash`.
    * It decodes the RLP data to load the Account object into the StateDB cache.
2. Execute Opcode `SSTORE(key, value)`:
    * The EVM identifies the specific Storage Trie belonging to this account.
    * It computes `slotHash = Keccak256(key)`.
    * It locates the value in the cached Storage Trie.
    * Mutation: It updates the value in the "Dirty" dictionary in memory.

Note: No hashing happens yet. The trie structure (nodes) is effectively "broken" in memory because the parent hashes no longer match the child data.

#### Merkle Patricia Trie Update

At the end of the transaction batch (or block building), the StateDB must "commit" these changes to generate the new Root. This involves structural trie manipulation.

### Proposer Computer Signing

To avoid one proposer computer overly burdened by block construction, builder computer actually just sent `ExecutionPayloadHeader` (the hash/summary of the transactions) to the proposer computer.

The proposer signs the `ExecutionPayloadHeader` indicating that the builder bid block is "locked", hereby maintaining the block chain not forked.

### Ethereum Query Flow for Account Proof

When a user (e.g., Alice, and `Keccak256(Alice)=0xA1F...`) requests balance proof of an account A, the Ethereum node (e.g., Geth) checks the LATEST block and retrieves `stateRoot`. Let's say the Block Header logged `stateRoot -> 0xROOT_888`.

Starting at `stateRoot -> 0xROOT_888` as tree root, Merkle Patricia Tree (MPT) search starts with the path `0xA1F...`.

### Ethereum Query Flow for Transaction Detail

Ethereum node (e.g., Geth) needs to traverse ALL blocks to find ALL transactions under an account.
A standard Ethereum node does NOT maintain a database index that maps `Address -> [List of Transactions]`.

#### The Etherscan/MetaMask Solution: Hosting An Off-Chain Lookup DB

Etherscan/MetaMask build a massive External Indexing Layer (usually a standard SQL database or specialized software).
They have a script that listens to every new block from Geth, and write data into a centralized database (like PostgreSQL).

## Ethereum Variant Chains

* ETH (Ethereum Mainnet - Proof of Stake)

In 2022, Ethereum upgraded ("The Merge") to Proof of Stake. This removed miners entirely. You cannot mine ETH with a GPU anymore; you "stake" coins to validate transactions.

* ETHW (Ethereum PoW)

When ETH switched to PoS, miners were left with billions of dollars in hardware and nowhere to go. A group of miners "forked" (copied) Ethereum right before the switch and removed the update. This allowed them to keep mining the old way.
Purpose: To give miners a way to keep using their equipment.

* ETC (Ethereum Classic)

This is actually the original Ethereum. In 2016, Ethereum was hacked (The DAO Hack). The developers decided to rewrite history to reverse the hack (creating today's ETH). Some people refused to rewrite history, staying on the original chain, which became ETC.
