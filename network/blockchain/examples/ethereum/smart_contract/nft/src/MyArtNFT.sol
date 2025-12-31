// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract MyArtNFT is ERC721URIStorage, Ownable {
    uint256 private _nextTokenId;

    // Constructor sets the name and symbol of your NFT collection
    constructor() ERC721("MyArtNFT", "MAC") Ownable(msg.sender) {}

    // The function to create (mint) the NFT
    // recipient: Wallet address receiving the NFT
    // tokenURI: The IPFS URL of the metadata.json (ipfs://<METADATA_CID>)
    function mintNFT(address recipient, string memory tokenURI)
        public
        onlyOwner
        returns (uint256)
    {
        uint256 tokenId = _nextTokenId++;
        _mint(recipient, tokenId);
        _setTokenURI(tokenId, tokenURI);

        return tokenId;
    }
}