source .env
# PRIVATE_KEY=...
# RPC_URL=https://rpc.ankr.com/eth/...
# CHAIN_ID=1
# ETHERSCAN_API_KEY=...

forge install OpenZeppelin/openzeppelin-contracts

forge script script/Deploy.s.sol \
    --rpc-url "$RPC_URL" \
    --chain-id "$CHAIN_ID" \
    --broadcast \
    -vvvv

# get from `network\blockchain\examples\ethereum\smart_contract\yuqi_coin\broadcast\Deploy.s.sol\1\run-latest.json`
forge verify-contract 0x2cca558caae41479dd5dfbd45d656bb54ab6c9c8 \
  src/YuqiCoin.sol:YuqiCoin \
  --chain-id 1 \
  --etherscan-api-key $ETHERSCAN_API_KEY \
  --constructor-args $(cast abi-encode "constructor(uint256)" 1000000000)