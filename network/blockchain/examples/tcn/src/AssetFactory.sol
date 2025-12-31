// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./AssetToken.sol";

contract AssetFactory {
    event AssetCreated(address indexed tokenAddress, string symbol);
    event SupplyChanged(address indexed tokenAddress, address indexed target, uint256 amount, string type_);

    mapping(bytes32 => address) private _tokenBySymbol;

    // Create a new independent ERC20 token (e.g., a new Bond issuance)
    function issueNewAsset(string memory name, string memory symbol) external returns (address) {
        (AssetToken token, bool created) = _getOrCreateAsset(name, symbol);
        require(created, "TCN: token exists");
        return address(token);
    }

    // Create and immediately distribute supply so explorers show balances
    function issueNewAssetWithDistribution(
        string memory name,
        string memory symbol,
        address[] calldata targets,
        uint256[] calldata amounts
    ) external returns (address) {
        require(targets.length == amounts.length, "TCN: length mismatch");
        (AssetToken token, ) = _getOrCreateAsset(name, symbol);
        for (uint256 i = 0; i < targets.length; ++i) {
            if (targets[i] == address(0) || amounts[i] == 0) {
                continue; // skip invalid entries instead of reverting entire call
            }
            token.mint(targets[i], amounts[i]);
            emit SupplyChanged(address(token), targets[i], amounts[i], "MINT");
        }
        return address(token);
    }

    // Dynamic Minting (Only accessible if this Factory owns the token)
    function mintAsset(address token, address to, uint256 amount) external {
        AssetToken(token).mint(to, amount);
        emit SupplyChanged(token, to, amount, "MINT");
    }

    // Dynamic Burning
    function burnAsset(address token, address from, uint256 amount) external {
        AssetToken(token).burn(from, amount);
        emit SupplyChanged(token, from, amount, "BURN");
    }

    function _getOrCreateAsset(string memory name, string memory symbol)
        private
        returns (AssetToken token, bool created)
    {
        bytes32 key = keccak256(bytes(symbol));
        address existing = _tokenBySymbol[key];
        if (existing != address(0)) {
            return (AssetToken(existing), false);
        }

        token = new AssetToken(name, symbol, address(this));
        _tokenBySymbol[key] = address(token);
        emit AssetCreated(address(token), symbol);
        created = true;
    }

    function getTokenBySymbol(string memory symbol) external view returns (address) {
        return _tokenBySymbol[keccak256(bytes(symbol))];
    }
}