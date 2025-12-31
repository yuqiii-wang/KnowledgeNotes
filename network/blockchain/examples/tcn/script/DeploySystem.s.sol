// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/AssetFactory.sol";
import "../src/DvPSettlement.sol";

contract DeploySystem is Script {
    function run() external {
        uint256 deployerKey = 0x8f2a55949038a9610f50fb23b5883af3b4ecb3c3bb792cbcefbd1542c692be63;
        vm.txGasPrice(0); // force zero-fee txs for private collateral network
        
        vm.startBroadcast(deployerKey);

        // AssetFactory factory = new AssetFactory();
        DvPSettlement dvp = new DvPSettlement();

        vm.stopBroadcast();

        console.log("AssetFactory:", address(factory));
        console.log("DvPSettlement:", address(dvp));
    }
}