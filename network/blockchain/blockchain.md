# Blockchain 

## Token

Tokens are a type of cryptocurrency that are used as a specific asset or represent a particular use on the blockchain. They are offered as monetary incentive to encourage blockchain miners to process transactions.

ERC-20 (Ethereum Request for Comments 20) is a Token Standard that implements an API for tokens within Smart Contracts.

### How to get ERC-20 Address (ERC-20 Token Transactions)

The address has to be a ERC-20 standard compatable address, and is exposed to the public Etherum blockchain. Sending coins to another wallet address needs paying a gas fee to miners as rewards to their assistance in processing the transaction.

When there's at least one transaction over a blockchain network, the address is registered on the network and maintained by miners.

Alternatively, you can use MEW (My Ether Wallet) to generate a ERC-20 standard compatable address. However, be aware that the address and passkey is user-custody, that means MEW does store it on their database.

## Smart Contract

A "smart contract" is simply a collection of code (its functions) and data (its state) that resides at a specific address on the Ethereum blockchain.

Smart contracts have a balance and they can send transactions over the network, run as a program, in which users can submits transactions.

An example of a vending machine where:
1. some init values such as init balances (`constructor` function)
2. a cupcake shop onwer provide physical cupcakes to customers (`refill` function)
3. customer pay the shop onwer by ETH coins (`purchase` function)
```js
pragma solidity 0.8.7;

contract VendingMachine {

    // Declare state variables of the contract
    address public owner;
    mapping (address => uint) public cupcakeBalances;

    // When 'VendingMachine' contract is deployed:
    // 1. set the deploying address as the owner of the contract
    // 2. set the deployed smart contract's cupcake balance to 100
    constructor() {
        owner = msg.sender;
        cupcakeBalances[address(this)] = 100;
    }

    // Allow the owner to increase the smart contract's cupcake balance
    function refill(uint amount) public {
        require(msg.sender == owner, "Only the owner can refill.");
        cupcakeBalances[address(this)] += amount;
    }

    // Allow anyone to purchase cupcakes
    function purchase(uint amount) public payable {
        require(msg.value >= amount * 1 ether, "You must pay at least 1 ETH per cupcake");
        require(cupcakeBalances[address(this)] >= amount, "Not enough cupcakes in stock to complete this purchase");
        cupcakeBalances[address(this)] -= amount;
        cupcakeBalances[msg.sender] += amount;
    }
}
```

## Coin Transactions

### Wei

Wei is the smallest denomination of ether—the cryptocurrency coin used on the Ethereum network. 

ether == $10^{18}$ wei == 1,000,000,000,000,000,000 == 1 ETH

### Transaction Signature

Ethereum Private Keys are 64 random hex characters or 32 random bytes.

The public key is derived from the private key using ECDSA.

The private key creates a signature. The public key verifies the signature.