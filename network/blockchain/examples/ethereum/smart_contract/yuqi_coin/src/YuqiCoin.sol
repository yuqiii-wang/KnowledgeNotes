// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

// --- The Token ---
contract YuqiCoin is ERC20 {
    constructor(uint256 initialSupply) ERC20("YuqiCoin", "YUQIC") {
        _mint(msg.sender, initialSupply * 10**decimals());
    }
}
