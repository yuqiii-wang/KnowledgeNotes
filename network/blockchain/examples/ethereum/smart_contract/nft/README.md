# Make NFT Guide

1. Upload Image: Upload `art.png` to Pinata. Copy the CID.
2. Create Metadata: Create `metadata.json`:

```json
{
  "name": "Foundry Art NFT",
  "description": "Minted with Forge and Cast",
  "image": "ipfs://<YOUR_IMAGE_CID_FROM_PINATA>"
}
```

3. Upload Metadata: Upload `metadata.json` as a file to Pinata.
4. Save the CID of `metadata.json`: This is Token URI. (e.g., `ipfs://QmPc...`)
