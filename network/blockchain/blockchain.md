# Blockchain 

## Basic Concepts: Token vs Transaction vs Block vs Coin

||Transaction (User Event)|Block (The Container)|Coin (Native Asset)|Token (Commercial Asset)|
|:---|:---|:---|:---|:---|
|What is it?|A single request to move funds or execute code.|A bundle of transactions "sealed" together.|The native "money" of the blockchain protocol.|Representative of commercial assets, e.g., USDT, USDC|
|Created By|An individual user (via Private Key).|The Network Validators (Consensus).|The Protocol (Hardcoded Math).|A Smart Contract (Company/Developer).|
|Example|"Send 5 USDC to Bob."|Block #19,283,744.|ETH (Ethereum), SOL (Solana).|USDC, LINK, SHIB, UNI.|

### What is behind trading in blockchain

1. John purchased 1000 USDC paid $1000 USD deposited to Circle (who issued USDC) bank account.
2. Circle backend triggers a smart contract execution that logs 1000 USDC to John

The USDC smart contract is publicly visible.
The core logic writes in `FiatTokenV2`, which is just a ledger created/updated dedicated to this customer John.

```js
// This logic is what the Proxy "runs" using its own storage
function mint(address _to, uint256 _amount) external whenNotPaused onlyMinters returns (bool) {
    require(_to != address(0), "Cannot mint to zero address");
    require(_amount > 0, "Amount must be greater than 0");

    // Check if the minter has enough "allowance" from Circle to create new coins
    uint256 mintingAllowedAmount = minterAllowances[msg.sender];
    require(_amount <= mintingAllowedAmount, "Exceeds minting allowance");

    // Update the ledger (The Data Storage part)
    totalSupply_ = totalSupply_.add(_amount);
    balances[_to] = balances[_to].add(_amount); // This gives John his 1000 USDC
    minterAllowances[msg.sender] = mintingAllowedAmount.sub(_amount);

    emit Mint(msg.sender, _to, _amount);
    emit Transfer(address(0), _to, _amount);
    return true;
}
```

About this smart contract execution, it can be considered as a *transaction*.
This transaction is appended in a *block* that on certain conditions, e.g., block processing interval of 10 mins, will be finalized by ethereum nodes and broadcast to all nodes.

This smart contract consumes *native coin* (ETH) since the whole ecosystem run on ethereum where ETH represents the computation and storage resource.
The said 1000 *USDC* is just a value on company Circle smart contract.

3. John starts using 1000 USDC to buy other crypto coins.

    Trading for ETH (or other same-chain coins):

    -> Use a Decentralized Exchange (DEX) like Uniswap.

    -> Must already have a small amount of ETH in wallet to pay the "Gas Fee" (the miners/validators) to process the transaction.

    Trading for BTC and DOGE (Cross-Chain):

    -> There are bridge apps that lock USDC and hook up the same worth coins and proceed trading. This could take minutes ot even hours to complete.

    Alternative to cross-chain trading:

    -> There are "wrapped coins" that represent John as the owner but held in custody by the company who issued that wrapped coins

4. John enjoys his investment in blockchain; company Circle can use John's $1000 to invest US Treasure bonds (claimed by Circle that every USDC is backed by US Treasure bonds OR JUST USD cash) to earn coupon money.

## Smart Contract

A "smart contract" is simply a collection of code (its functions) and data (its state) that resides at a specific address on the Ethereum blockchain.

A very common example is issuing a new coin.

```js
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract TopUpCoin is ERC20 {

    constructor() ERC20("TopUpCoin", "TUC") {
        // Mint 1000 tokens to the person who deploys it
        _mint(msg.sender, 1000); 
    }

    function topUp(uint256 amount) public {
        // _mint is an internal ERC20 function that increases 
        // the total supply and adds to the user's balance.
        _mint(msg.sender, amount);
    }
}
```

where the persistent and ephemeral data are

* The persistent data and operation
    * The Ledger (`_balances`): A mapping that looks like `address` => `uint256`.
    * The Total Supply (`_totalSupply`): A `uint256` that tracks the total number of coins in existence.
    * Metadata (`_name` and `_symbol`): The strings `"YuqiCoin"` and `"YUQIC"`. These are stored in the contract's storage during the constructor and usually never change again.
    * Cost: When `_mint` is called, it writes to the `_balances` mapping. This is the most expensive part of the transaction (Gas-wise).
        In `constructor()` by default it deposits 1000 Yuqi coins, then user can `topUp` with a custom amount.
* Ephemeral Data (live in a session)
    * The Parameter (`initialSupply`)
    * Global Variables (`msg.sender`)
    * Intermediate Calculations: The math `initialSupply * 10**decimals()` happens in the Stack. 

```js
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/YuqiCoin.sol";

contract DeployScript is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerPrivateKey);

        // Deploy Token
        YuqiCoin token = new YuqiCoin(100);
        
        vm.stopBroadcast();
    }
}
```

where

* `vm.startBroadcast()` and `vm.stopBroadcast()`
    * `vm`: This is a special "Cheatcode" object available only in scripts and tests.
    * `startBroadcast(privateKey)`: This tells Foundry to use that specific private key to sign transactions. Any new contract or function call following this line will be treated as a real transaction to be sent to the blockchain.

### Common Smart Contract Syntax

* `require(condition, "error")`: Revert transaction if condition is false and return a message.
* `msg.sender`: The immediate caller address.
* `msg.value`: Ether (in wei) sent with the call.
* `keccak256(abi.encodePacked(...))`: Hashing function to generate bytes32 ids.
* `abi.encodePacked(...)`: Tight encoding used with `keccak256`.
* `mapping(Key => Value)`: Key-value storage
* `event` and `emit`: Declare and trigger events for off-chain listeners. Example: `event Registered(...)` and `emit Registered(...)`.
* `indexed (in event params)`: Makes a field searchable in logs (e.g., `bytes32 indexed id`).
* `modifier`: Reusable pre/post checks.
* `constructor()`: Runs once on deployment to initialize state.
* Function visibility keywords: `public`, `external`, `internal`, `private` (file uses `public` and `view`).
* Function mutability keywords:
    * `view`: reads state, does not modify it.
    * `pure`: does not read or modify state.
    * `payable`: accepts Ether.
    * (implicit) non-payable: default — rejects Ether.
* Data location
    * `memory`: creates a temporary, mutable copy that lives only for the function call — appropriate required a local copy or plan to mutate it. 
    * `calldata`: a non‑mutable `view` directly into the call data — cheapest (no copy) and recommended for external functions that only read the input. Example: `function register(string calldata data) external returns (bytes32)`.
    * `storage`: refers to persistent contract storage. 

Below is a more complex smart contract example that registers data ownership by binding data hash to the owner addr.

```js
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MiniRegistry {
    mapping(address => bytes32[]) public ownerItems; // owner -> list of ids
    mapping(bytes32 => string) public items;         // id -> data

    event Registered(bytes32 indexed id, address indexed owner);

    function register(string memory data) public returns (bytes32) {
        bytes32 id = keccak256(abi.encodePacked(msg.sender, block.timestamp, data));
        items[id] = data;
        ownerItems[msg.sender].push(id);
        emit Registered(id, msg.sender);
        return id;
    }
}
```

where

The storage data that will be persisted on blockchain is

* `ownerItems` (storage): `mapping(address => bytes32[])` — persistent ledger: for each owner address, a dynamic array of `bytes32` ids (each .push writes storage and costs gas).
* `items` (storage): `mapping(bytes32 => string)` — persistent mapping from id → stored string data.
* Contract balance: `address(this).balance` — Ether held by the contract is stored by the EVM account, not in your Solidity variables.
* Contract code & storage root: the deployed bytecode and all storage slots (including mappings/arrays) are persisted on-chain.

Smart contract itself runs on EVM, developer/user can input data, and see output to/from EVM via `event` and catching `emit` data.
The user-interaction data is

* `event`: declares a log type (`Registered`) that off-chain listeners can filter.
* `emit`: triggers the event log (`emit Registered(id, msg.sender);`).

### Smart Contract Invocation and Listening

ABI (Application Binary Interface) defines how to construct request.
It is created from the definition of smart contract.

Below is the decoded ABI (Application Binary Interface).

```js
const ABI = [
  {"inputs":[{"internalType":"string","name":"data","type":"string"}],"name":"register","outputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"stateMutability":"nonpayable","type":"function"},
  {"anonymous":false,"inputs":[{"indexed":true,"internalType":"bytes32","name":"id","type":"bytes32"},{"indexed":true,"internalType":"address","name":"owner","type":"address"}],"name":"Registered","type":"event"}
]
```

Having created/deployed the contract, developer receives `CONTRACT_ADDRESS`.
With `ABI`, developer prepares `ethers.Contract(CONTRACT_ADDRESS, ABI, signer);`, then `contract.registerWork(...)` can be dynamically defined from the deployed contract.

`receipt.logs` catches smart contract `emit` data.

Below is the full js application that interacts with EVM.

```js
const { ethers } = require("ethers");

const RPC_URL = process.env.RPC_URL;
const PRIVATE_KEY = process.env.PRIVATE_KEY;
const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS;

async function main() {
  const provider = new ethers.providers.JsonRpcProvider(RPC_URL);
  const signer = new ethers.Wallet(PRIVATE_KEY, provider);
  const contract = new ethers.Contract(CONTRACT_ADDRESS, ABI, signer);

  const tx = await contract.registerWork(
    "My Work",
    "desc",
    ["QmHash1"],
    ethers.constants.AddressZero,
    "Alice",
    { value: ethers.utils.parseEther("0.01") }
  );
  const receipt = await tx.wait();

  for (const log of receipt.logs) {
    try {
      const parsed = contract.interface.parseLog(log);
      if (parsed.name === "Registered") {
        console.log("Registered:", parsed.args);
      }
    } catch (e) {}
  }

  const count = await contract.getUserAssetCount(signer.address);
  console.log("assetCount:", count.toString());
}

main().catch(console.error);
```

### Wallet vs Smart Contract

|Fields|What it looks like for a Regular Wallet|What it looks like for a Contract|
|:---|:---|:---|
|Nonce|`0x5` (Count of transaction, 5 indicated sent 5 transactions)|`0x1` (Usually 1)|
|Balance|`0xde0b6b3a7640000` (1 ETH in hex)|Varies|
|CodeHash|`0xc5d2...ad27` (The hash of nothing)|The hash of the contract bytecode.|
|StorageRoot|`0x56e8...1662` (The hash of an empty tree)|The root hash of the contract’s data.|

The above info can be retrieved via `proof`.

```py
state = ankr_w3.eth.get_proof(MY_WALLET_ADDR, [], 'latest')

print(f"Account Proof for: {MY_WALLET_ADDR}")
print(f"Nonce:             {state.nonce}")
print(f"Balance:           {state.balance} Wei")
print(f"CodeHash:          {state.codeHash.hex()}")
print(f"StorageHash:       {state.storageHash.hex()}")
```

The nonce of regular wallet is used to prevent duplicate replays.

## Coin Transactions

### Wei

Wei is the smallest denomination of ether—the cryptocurrency coin used on the Ethereum network. 

ether == $10^{18}$ wei == 1,000,000,000,000,000,000 == 1 ETH

### Transaction Signature

Ethereum Private Keys are 64 random hex characters or 32 random bytes.

The public key is derived from the private key using ECDSA.

The private key creates a signature. The public key verifies the signature.

#### Gas Fee

Gas is the transaction fee paid to the chain computers to execute a transaction or smart contract.

Gas used in Ethereum (ETH) varies by transaction type:

* Simple ETH Transfer: 21,000 gas units.
* ERC-20 Token Transfer/Approval: Around 45,000 - 65,000 gas units.
* Smart Contract Interactions (DeFi, NFTs): Can range from 100,000 to 300,000+ gas units

#### Gas vs Coin

* Coin represents permission that a balance/wallet can make request to a blockchain; on detected zero coin in a wallet, EVM will immediately reject the request, even for the requests not required of consuming any computation nor storage resource. 
* Gas represents the computational effort of a smart contract.

#### Gas vs Transaction Fee

Given the log:

```txt
Gas used: 22160
Effective gas price (wei): 25193522
Transaction fee (wei): 558288447520
```

They mean

* `Gas used: 22160`: Measurement for computational effort, e.g., how many lines of bytecode to execute for a smart contract
* `Effective gas price (wei): 25193522`: To pay for each unit of Gas.
* `Transaction fee (wei): 558288447520`: $\text{Gas} \times \text{Gas Price} = \text{Transaction Fee}$

## MultiSig Wallet

Multi-signature wallets are a type of cryptocurrency wallet that requires **multiple keys** to unlock funds or approve transactions. 

## Proof of Truth: Consensus Mechanisms

In traditional IT architecture (a centralized system), the "truth" is whatever the system database holds.
In a decentralized system (blockchain), thousands of computers (nodes) must agree on the "truth" (which transactions are valid and in what order) without trusting each other.

Here is a list and explanation of the major consensus mechanisms.

### Proof of Work (PoW)

* To add a block to the chain, a computer (miner) must solve a continuously difficult mathematical puzzle (finding a hash). This requires massive amounts of electricity and hardware. The "work" refers to puzzle solving.
* Examples: Bitcoin, Dogecoin, Litecoin.

The puzzle result hash goes together with previous hash and this `user_input` to compute this block hash.

The pseudocode of one user input

```py
def make_one_blockchain_transaction(user_input, prev_hash):
    while True:
        puzzle_hash = puzzle()
        block_hash = hash(user_input, prev_hash, puzzle_hash)
        prev_hash = block_hash
    return user_input, prev_hash
```

Once a computer (miner) correctly solves the puzzle, it wins the right to add the next "Block" to the chain.
A computer (miner) can receive rewards (fractions of transaction fee) on processing/completing a transaction.

|Coin|Mechanism|Block Reward (The Jackpot)|
|:---|:---|:---|
|Bitcoin|Proof of Work|3.125 BTC|
|Dogecoin|Proof of Work|10,000 DOGE|

#### The Puzzle Detail in PoW (BitCoin as The Example)

1. A determined hash target

The Bitcoin protocol is programmed to ensure that, on average, one block is found every 10 minutes.

The network doesn't change the target for every block. It waits for 2,016 blocks to pass (which takes almost exactly 2 weeks if blocks come every 10 minutes).
When the 2,016th block is reached, every node in the world looks at its own copy of the blockchain and performs a calculation.

2. Find the value from the formula

$$
\text{SHA256}(\text{SHA256}(\text{Block Header})) < \text{Target}
$$

The "Block Header" is exactly 80 bytes of data. It is a concatenation of 6 fields:

|Field|Size|Description|
|:---|:---|:---|
|Version|4 bytes|Protocol version number.|
|HashPrevBlock|32 bytes|The SHA256 hash of the previous block header.|
|HashMerkleRoot|32 bytes|The "fingerprint" of all transactions in this block.|
|Time|4 bytes|Current Unix timestamp.|
|Bits (nBits)|4 bytes|A compressed version of the Target.|
|Nonce|4 bytes|The number the miner changes to get a new hash.|

#### The 51% Attack Prevention

To "cheat" or hack Bitcoin, the hacker computer would need to control more than 51% of the total computing power on Earth.
The cost of the electricity and hardware needed to do this is so high that it’s practically impossible.

### Proof of Stake (PoS)

* Instead of miners, there are Validators. To become a validator, user must lock up ("stake") a large amount of the cryptocurrency (e.g., 32 ETH) into a smart contract.
* The network randomly selects a validator to propose the next block. If the validator acts honestly, they get a reward. If they try to cheat (validate bad blocks, computer is service down), the network "slashes" (destroys) their staked money.
* Pros: Energy efficient (99.9% less energy than PoW); lower barrier to entry (no hardware farms needed).
* Cons: "Rich get richer" criticism; complex implementation (requires the Beacon Client).
* Examples: Ethereum (current), Cardano, Solana.

where, by 2025, for example, Ethereum needs Execution Client (e.g., Geth) and Beacon Client (e.g., Consensus).

#### The Execution and Beacon Client

* Execution Client (e.g., Geth): The worker. It deals with transactions, smart contracts, and balances. It is heavy on computation.
* Beacon Client (e.g., Consensus): The manager. It deals with people (validators), time, and agreement. It is heavy on networking and rules.

#### Block Process Flow under PoS

### Proof of Authority (PoA)

* There is no mining and no money at stake. Instead, a small group of pre-approved identities (*Signers*) has the right to create blocks.
* Pros: Extremely fast; zero transaction costs (if desired); perfect for private networks.
* Cons: Not decentralized. It relies on trusting specific humans/entities.
* Examples: Testnets (Sepolia, Goerli), Private Corporate Blockchains (JP Morgan’s Quorum).

#### Consensus Implementation and Standards

* Raft: A "Leader" (Minter) is elected. They propose blocks and everyone else follows. It is extremely fast (no empty blocks) but assumes no one is trying to "hack" the system (Crash Fault Tolerant).
* IBFT (Istanbul Byzantine Fault Tolerance): A group of validators takes turns proposing blocks. They must reach a 2/3 majority vote to finalize a block. This is slightly slower but can survive malicious nodes (Byzantine Fault Tolerant).

## The 5-Layer Arch of Blockchain

* Summary Table: Mapping Blockchain to OSI

|Blockchain Layer|Maps to OSI Layer|Why?|
|:---|:---|:---|
|Application Layer|Layers 5, 6, 7|This is the UI, the encryption, and the session management.|
|Execution/Socket Layer|(None / Upper L4)|OSI has no concept of a "Virtual Machine" to run global code.|
|Consensus Layer|(None)|Traditional networking doesn't require "Agreement" to move data.|
|Network (P2P) Layer|Layer 3 & 4|Uses IP and TCP/UDP to discover peers and broadcast blocks.|
|Infrastructure Layer|Layer 1 & 2|The literal wires and computers.|

## Inter-Planetary File System (IPFS)

IPFS (Inter-Planetary File System) is a peer-to-peer distributed file system that is used primarily for data that can’t be stored on a blockchain.
