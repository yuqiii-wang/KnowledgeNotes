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
|Reth (Rust Ethereum)|Rust|revm <br>(Modular & Extremely Fast)|MDBX|Flat KV <br>(1 Disk Read per lookup)|Max Performance <br>(Fastest sync & execution)|
|Nethermind|C# (.NET)|JIT Compiler <br>(Just-In-Time optimization)|RocksDB|Paprika <br>(Custom Optimized Trie)|Corporate / Windows <br>(Great for institutional setups)|
|Erigon|Go|JumpTable <br>(Forked from Geth)|MDBX|Flat KV <br>(Separates Data from Trie)|Data Analytics <br>(Efficient Archive Nodes)|

P.S., since v1.14, Geth separates execution and consensus components, while in v1.13, a lightweight consensus engine *Clique* in embedded.

### The EVM Consensus Components

The Consensus Client handles the Beacon Chain. It manages the validators, tracks the logic of who votes for what block.

|Client|Language|Developed By|Resource Usage|Key Strength|Best For|
|:---|:---|:---|:---|:---|:---|
|Prysm|Go|Prysmatic Labs (Offchain Labs)|Medium/High|User Experience. Great documentation, widely used.|Beginners & Geth Users|
|Lighthouse|Rust|Sigma Prime|Low|Performance. Extremely stable, safe, and fast.|Power Users & Stakers|
|Teku|Java|ConsenSys|High (RAM)|Enterprise. Built for institutional setups.|Banks & Institutions|

## Ethereum Network Types

In the Ethereum ecosystem, the software (EVM) is identical across all versions, but the data (ledger) and the peers (network) are different.

|Feature|Mainnet (Production)|Testnets (Sepolia / Holesky)|Private Networks (Quorum)|
|:---|:---|:---|:---|
|Purpose|Real Economy & Business|Development & Debugging|Enterprise & Privacy|
|Currency|Real ETH ($)|Fake ETH (Worth $0)|No Value (or Custom Token)|
|Consensus|Proof of Stake (PoS)|PoS (Permissioned)|Proof of Authority (PoA)|
|Security|Global ($60B+ staked)|Low (Centralized validators)|High (Walled Garden)|
|Access|Public (Anyone)|Public (Anyone)|Restricted (Invite Only)|
