# Tokenized Collateral Network - DvP Trading Simulation

To start Quorum blockchain

```sh
npx quorum-dev-quickstart

# to start
./run.sh

# to remove
./remove.sh
```

Successful run will output

```txt
*************************************
Quorum Dev Quickstart
*************************************
----------------------------------
List endpoints and services
----------------------------------
JSON-RPC HTTP service endpoint                 : http://localhost:8545
JSON-RPC WebSocket service endpoint            : ws://localhost:8546
Web block explorer address                     : http://localhost:25000/explorer/nodes
Chainlens address                              : http://localhost:8081/
Blockscout address                             : http://localhost:26000/
Prometheus address                             : http://localhost:9090/graph
Grafana address                                : http://localhost:3000/d/a1lVy7ycin9Yv/goquorum-overview?orgId=1&refresh=10s&from=now-30m&to=now&var-system=All
Collated logs using Grafana and Loki           : http://localhost:3000/d/Ak6eXLsPxFemKYKEXfcH/quorum-logs-loki?orgId=1&var-app=quorum&var-search=

For more information on the endpoints and services, refer to README.md in the installation directory.
****************************************************************
```

Deploy smart contract.

```sh
forge build

NO_PROXY=127.0.0.1,localhost \
forge script script/DeploySystem.s.sol:DeploySystem \
  --rpc-url "${RPC_URL:-http://127.0.0.1:8545}" \
  --broadcast \
  --legacy \
  --gas-price 0 \
  --slow
```

From `boradcast/DeploySystem.s.sol/.../run-latest.json` find the addr of `AssetFactory`.

```json
[
  {
    ...
    "contractName": "AssetFactory",
    "contractAddress": "0xd6a7c915066e17ba18024c799258c8a286ffbc00",
  },
  {
    ...
    "contractName": "DvPSettlement",
    "contractAddress": "0x2f9f1bc72ff3bf57849d83f25348f90fb8056f75",
  },
  ...
]
```

Create two wallets; the wallet addrs are in `wallet-a.txt` and `wallet-b.txt`

```sh
KEYSTORE_DIR=wallet-keystore
mkdir -p "$KEYSTORE_DIR"

# Wallet A
cast wallet new \
  --password "$KEYSTORE_DIR" \
  | tee wallet-a.txt

# Wallet B
cast wallet new \
  --password "$KEYSTORE_DIR" \
  | tee wallet-b.txt
```

Give some assets to the two wallets.
The success results shall be present on `http://localhost:26000/tokens`.

```sh
export RPC_URL=http://127.0.0.1:8545
export FACTORY=0x42699a7612a82f1d9c36148af9c77354759b210b
export DVP=0xa50a51c09a5c451c52bb714527e1974b686d8e77
export PK=0x8f2a55949038a9610f50fb23b5883af3b4ecb3c3bb792cbcefbd1542c692be63
export NO_PROXY=127.0.0.1,localhost
export RECIPIENT_A=0x4238174A4B117C42662c1452312671069C77F54a  # Wallet A
export RECIPIENT_B=0x19232be5c53625320Cbd92c5C792BC5c7C3F5D0a  # Wallet B

# Shell-safe arrays cast can parse for address[] / uint256[]
RECIPIENTS="[$RECIPIENT_A,$RECIPIENT_B]"
CASH_AMOUNTS="[150000000000000000000000,150000000000000000000000]"
BOND_AMOUNTS="[1000000000000000000000,1000000000000000000000]"

# Dummy ISIN → ticker map (issuer coupon maturity_date)
declare -A ISIN_TICKER_MAP=(
  [US123456CF90]="BABA 2.50% 2028-06-15"
  [US0987654321]="UST 3.10% 2030-12-01"
  [FR0000001111]="OAT 1.80% 2027-09-20"
  [DE7700004321]="BMW 3 3/4 09/35"
  [XS3100004366]="BENZ 3 1/4 04/35"
)

declare -A CCY_CASH_MAP=(
  [USD]="USD Cash"
  [GBP]="GBP Cash"
  [EUR]="EUR Cash"
)

# Issue a USD cash token and immediately distribute to both wallets
for CCY in USD GBP EUR; do
  CCY_CASH=${CCY_CASH_MAP[$CCY]}
  if [ -z "$CCY_CASH" ]; then
    echo "no currency found for $CCY, skipping"
    continue
  fi
  TX_HASH=$(cast send "$FACTORY" \
    "issueNewAssetWithDistribution(string,string,address[],uint256[])" \
    "$CCY_CASH" \
    "$CCY" \
    "$RECIPIENTS" \
    "$CASH_AMOUNTS" \
    --rpc-url "$RPC_URL" \
    --gas-price 0 \
    --legacy \
    --private-key "$PK")
  echo "cash issue tx: $TX_HASH"
done

# Issue multiple ISIN bonds, splitting supply across Wallet A and B
for ISIN in US123456CF90 US0987654321 FR0000001111 DE7700004321 XS3100004366; do
  TICKER=${ISIN_TICKER_MAP[$ISIN]}
  if [ -z "$TICKER" ]; then
    echo "no ticker found for $ISIN, skipping"
    continue
  fi
  TX=$(cast send "$FACTORY" \
    "issueNewAssetWithDistribution(string,string,address[],uint256[])" \
    "$TICKER" \
    "$ISIN" \
    "$RECIPIENTS" \
    "$BOND_AMOUNTS" \
    --rpc-url "$RPC_URL" \
    --gas-price 0 \
    --legacy \
    --private-key "$PK")
  echo "bond issue tx ($ISIN): $TX"
done
```

Conduct a DvP trade, wallet A trade 50 units of two bonds US123456CF90 and US0987654321 for $10,100 = $101 * (50 + 50) from wallet B.

```sh
export PK_SELLER=$(cast wallet decrypt-keystore 'C:\Users\yuqi\Documents\KnowledgeNotes\network\blockchain\examples\tcn\wallet-keystore\ef921ccd-658c-44fc-bbc2-5ded16533637' | grep -Eo '0x[0-9a-fA-F]{64}') # Wallet A
export PK_BUYER=$(cast wallet decrypt-keystore 'C:\Users\yuqi\Documents\KnowledgeNotes\network\blockchain\examples\tcn\wallet-keystore\a27bebb0-4758-4fca-a238-580ba1ba6e91' | grep -Eo '0x[0-9a-fA-F]{64}') # Wallet B


# Look up token addresses (requires factory redeployed with getTokenBySymbol)
export BOND_US123456CF90=$(cast call "$FACTORY" "getTokenBySymbol(string)(address)" "US123456CF90")
export BOND_US0987654321=$(cast call "$FACTORY" "getTokenBySymbol(string)(address)" "US0987654321")
export USD_TOKEN=$(cast call "$FACTORY" "getTokenBySymbol(string)(address)" "USD")

# Amounts are already scaled to 18 decimals, so no `ether` suffix is needed on this private network
export BOND_TRADE_AMOUNTS="[500000000000000000000,500000000000000000000]"   # 50 units each
export USD_TRADE_AMOUNTS="[10100000000000000000000]"                         # 10,100 units total
export TRADE_NONCE=1

cast send "$BOND_US123456CF90" "approve(address,uint256)" \
  "$DVP" 500000000000000000000 \
  --rpc-url "$RPC_URL" \
  --gas-price 0 \
  --legacy \
  --private-key "$PK_SELLER"

cast send "$BOND_US0987654321" "approve(address,uint256)" \
  "$DVP" 500000000000000000000 \
  --rpc-url "$RPC_URL" \
  --gas-price 0 \
  --legacy \
  --private-key "$PK_SELLER"

cast send "$USD_TOKEN" "approve(address,uint256)" \
  "$DVP" 10100000000000000000000 \
  --rpc-url "$RPC_URL" \
  --gas-price 0 \
  --legacy \
  --private-key "$PK_BUYER"

cast call "$BOND_US123456CF90" "allowance(address,address)(uint256)" "$RECIPIENT_A" "$DVP"
cast call "$BOND_US0987654321" "allowance(address,address)(uint256)" "$RECIPIENT_A" "$DVP"
cast call "$USD_TOKEN" "allowance(address,address)(uint256)" "$RECIPIENT_B" "$DVP"


# Both parties authorize the same trade parameters (multisig-style approval)
cast send "$DVP" \
  "authorizeTrade(address,address,address[],uint256[],address[],uint256[],uint256)" \
  "$RECIPIENT_A" \
  "$RECIPIENT_B" \
  "[$BOND_US123456CF90,$BOND_US0987654321]" \
  "$BOND_TRADE_AMOUNTS" \
  "[$USD_TOKEN]" \
  "$USD_TRADE_AMOUNTS" \
  "$TRADE_NONCE" \
  --rpc-url "$RPC_URL" \
  --gas-price 0 \
  --legacy \
  --private-key "$PK_SELLER"

cast send "$DVP" \
  "authorizeTrade(address,address,address[],uint256[],address[],uint256[],uint256)" \
  "$RECIPIENT_A" \
  "$RECIPIENT_B" \
  "[$BOND_US123456CF90,$BOND_US0987654321]" \
  "$BOND_TRADE_AMOUNTS" \
  "[$USD_TOKEN]" \
  "$USD_TRADE_AMOUNTS" \
  "$TRADE_NONCE" \
  --rpc-url "$RPC_URL" \
  --gas-price 0 \
  --legacy \
  --private-key "$PK_BUYER"

# Execute DvP: Wallet A sells 50 units of each bond to Wallet B for 10,100 USD
cast send "$DVP" \
  "executeAtomicTrade(address,address,address[],uint256[],address[],uint256[],uint256)" \
  "$RECIPIENT_A" \
  "$RECIPIENT_B" \
  "[$BOND_US123456CF90,$BOND_US0987654321]" \
  "$BOND_TRADE_AMOUNTS" \
  "[$USD_TOKEN]" \
  "$USD_TRADE_AMOUNTS" \
  "$TRADE_NONCE" \
  --rpc-url "$RPC_URL" \
  --gas-price 0 \
  --legacy \
  --private-key "$PK_SELLER"
```

## Making the DvP trade private in Quorum

Privacy transactions use node addrs rather than wallet addrs.
Once added `privateFrom` and `privateFor`, transaction detail can be only read by the specified nodes. Other nodes can only see contract execution without execution detail.

The node info can be found in `smart_contracts/scripts/keys.js` (for quorum).

```sh
# Additional privacy-specific variables on top of the public DvP exports
export RPC_URL=http://127.0.0.1:20000 
export PRIVATE_FROM='BULeR8JyUWhiuuCMU/HLA0Q5pzkYT+cHII3ZKBey3Bo='  # seller node enclave key
export PRIVATE_FOR='["BULeR8JyUWhiuuCMU/HLA0Q5pzkYT+cHII3ZKBey3Bo="]'  # buyer enclave + optional notary so only those parties see the trade
export NODE_FROM="0xf0e2db6c8dc6c681bb5d6ad121a107f300e9b2b5"
export NODE_TO="0xca843569e3427144cead5e4d5999a3d0ccf92b8e"


cast rpc --rpc-url ${RPC_URL} \
  personal_importRawKey 8bbbb1b345af56b560a5b20bd4b0ed1cd8cc9958a16262bc75118453cb546df7 ""
cast rpc --rpc-url ${RPC_URL} \
  personal_importRawKey 4762e04d10832808a0aebdaa79c12de54afbe006bfffd228b3abcc494fe986f9 ""
cast rpc --rpc-url ${RPC_URL} \
  personal_importRawKey 61dced5af778942996880120b303fc11ee28cc8e5036d2fdff619b5675ded3f0 ""

cast rpc --rpc-url ${RPC_URL} \
  personal_unlockAccount 0xf0e2db6c8dc6c681bb5d6ad121a107f300e9b2b5 "" 0
cast rpc --rpc-url ${RPC_URL} \
  personal_unlockAccount 0xca843569e3427144cead5e4d5999a3d0ccf92b8e "" 0
cast rpc --rpc-url ${RPC_URL} \
  personal_unlockAccount 0x0fbdc686b912d7722dc86510934589e0aaf3b55a "" 0

cast rpc --rpc-url ${RPC_URL} eth_accounts

build_payload() {
  local FROM=$1 DATA=$2
  cat <<JSON
{
  "jsonrpc": "2.0",
  "method": "eth_sendTransaction",
  "params": [{
    "from": "$FROM",
    "to": "$DVP",
    "gas": "0xf4240",
    "gasPrice": "0x0",
    "value": "0x0",
    "data": "$DATA",
    "privateFrom": "$PRIVATE_FROM",
    "privateFor": $PRIVATE_FOR,
    "restriction": "restricted"
  }],
  "id": 7
}
JSON
}

# 1) Seller privately authorizes the trade
AUTH_DATA=$(cast calldata "authorizeTrade(address,address,address[],uint256[],address[],uint256[],uint256)" \
  "$RECIPIENT_A" \
  "$RECIPIENT_B" \
  "[$BOND_US123456CF90,$BOND_US0987654321]" \
  "$BOND_TRADE_AMOUNTS" \
  "[$USD_TOKEN]" \
  "$USD_TRADE_AMOUNTS" \
  "$TRADE_NONCE")

curl -s "$RPC_URL" \
  -H "Content-Type: application/json" \
  -d "$(build_payload "$NODE_FROM" "$AUTH_DATA")"

# 2) Buyer authorizes the same terms privately
curl -s "$RPC_URL" \
  -H "Content-Type: application/json" \
  -d "$(build_payload "$NODE_TO" "$AUTH_DATA")"

# 3) Seller executes the DvP privately with identical calldata from the public section
EXEC_DATA=$(cast calldata "executeAtomicTrade(address,address,address[],uint256[],address[],uint256[],uint256)" \
  "$NODE_FROM" \
  "$NODE_TO" \
  "[$BOND_US123456CF90,$BOND_US0987654321]" \
  "$BOND_TRADE_AMOUNTS" \
  "[$USD_TOKEN]" \
  "$USD_TRADE_AMOUNTS" \
  "$TRADE_NONCE")

curl -s "$RPC_URL" \
  -H "Content-Type: application/json" \
  -d "$(build_payload "$NODE_FROM" "$EXEC_DATA")"
```
