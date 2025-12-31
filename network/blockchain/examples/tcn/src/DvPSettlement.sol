// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "forge-std/interfaces/IERC20.sol";

contract DvPSettlement {
    event TradeAuthorized(bytes32 indexed tradeHash, address indexed approver);
    event TradeExecuted(address indexed seller, address indexed buyer, uint256 indexed nonce, uint256 timestamp);

    bool private _locked;

    struct TradeApproval {
        bool sellerApproved;
        bool buyerApproved;
        bool executed;
    }

    mapping(bytes32 => TradeApproval) private _tradeApprovals;

    modifier nonReentrant() {
        require(!_locked, "DvP: reentrant call");
        _locked = true;
        _;
        _locked = false;
    }

    function authorizeTrade(
        address seller,
        address buyer,
        address[] calldata sellerTokens,
        uint256[] calldata sellerAmounts,
        address[] calldata buyerTokens,
        uint256[] calldata buyerAmounts,
        uint256 nonce
    ) external {
        _validateLegs(seller, buyer, sellerTokens, sellerAmounts, buyerTokens, buyerAmounts);
        require(msg.sender == seller || msg.sender == buyer, "DvP: unauthorized approver");

        bytes32 tradeHash = _computeTradeHash(seller, buyer, sellerTokens, sellerAmounts, buyerTokens, buyerAmounts, nonce);
        TradeApproval storage approval = _tradeApprovals[tradeHash];
        require(!approval.executed, "DvP: trade executed");

        if (msg.sender == seller) {
            approval.sellerApproved = true;
        } else {
            approval.buyerApproved = true;
        }

        emit TradeAuthorized(tradeHash, msg.sender);
    }

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

    function _validateLegs(
        address seller,
        address buyer,
        address[] calldata sellerTokens,
        uint256[] calldata sellerAmounts,
        address[] calldata buyerTokens,
        uint256[] calldata buyerAmounts
    ) private pure {
        require(seller != address(0) && buyer != address(0), "DvP: invalid party");
        require(sellerTokens.length == sellerAmounts.length, "Seller data mismatch");
        require(buyerTokens.length == buyerAmounts.length, "Buyer data mismatch");
        require(sellerTokens.length > 0 && buyerTokens.length > 0, "DvP: empty legs");
    }

    function _computeTradeHash(
        address seller,
        address buyer,
        address[] calldata sellerTokens,
        uint256[] calldata sellerAmounts,
        address[] calldata buyerTokens,
        uint256[] calldata buyerAmounts,
        uint256 nonce
    ) private pure returns (bytes32) {
        return keccak256(abi.encode(seller, buyer, sellerTokens, sellerAmounts, buyerTokens, buyerAmounts, nonce));
    }

    function _settleSellerLeg(
        address seller,
        address buyer,
        address[] calldata sellerTokens,
        uint256[] calldata sellerAmounts
    ) private {
        for (uint256 i = 0; i < sellerTokens.length; i++) {
            address token = sellerTokens[i];
            uint256 amount = sellerAmounts[i];
            require(token != address(0), "DvP: invalid seller token");
            require(amount > 0, "DvP: invalid seller amt");
            require(IERC20(token).transferFrom(seller, buyer, amount), "Bond tx failed");
        }
    }

    function _settleBuyerLeg(
        address buyer,
        address seller,
        address[] calldata buyerTokens,
        uint256[] calldata buyerAmounts
    ) private {
        for (uint256 i = 0; i < buyerTokens.length; i++) {
            address token = buyerTokens[i];
            uint256 amount = buyerAmounts[i];
            require(token != address(0), "DvP: invalid buyer token");
            require(amount > 0, "DvP: invalid buyer amt");
            require(IERC20(token).transferFrom(buyer, seller, amount), "Cash tx failed");
        }
    }
}