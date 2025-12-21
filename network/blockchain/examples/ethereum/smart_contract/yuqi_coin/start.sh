# curl -L https://foundry.paradigm.xyz | bash
# foundryup

source .env

forge script script/Deploy.s.sol \
    --rpc-url $RPC_URL \
    --broadcast \
    --etherscan-api-key $ETHERSCAN_API_KEY \
    -vvvv