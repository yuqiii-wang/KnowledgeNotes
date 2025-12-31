// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/MyArtNFT.sol";

contract DeployNFT is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");

        vm.startBroadcast(deployerPrivateKey);

        MyArtNFT nft = new MyArtNFT();

        // 4. (Optional) Mint the first NFT immediately during deployment
        // Replace with your Wallet Address and Metadata CID
        string memory uri = "ipfs://YOUR_METADATA_CID_FROM_STEP_1";
        nft.mintNFT("0xYourWalletAddressHere...", uri);

        // 5. Stop recording
        vm.stopBroadcast();
    }
}