from datetime import datetime
from web3 import Web3  # ankr_web3 is Web3-compatible

from ankr import AnkrWeb3
from ankr.types import GetAccountBalanceRequest
from eth_account import Account

from _config import (
    ETH_PROVIDER_API_KEY,
    MY_WALLET_ADDR,
    MY_ANOTHER_WALLET_ADDR,
    TRANSACTION_20251220_HASH,
    PRIVATE_KEY
)

ankr_w3 = AnkrWeb3(ETH_PROVIDER_API_KEY)

print("-" * 20)

# 1. Unlock account using private key
assets = ankr_w3.token.get_account_balance(
    request=GetAccountBalanceRequest(
        walletAddress=MY_WALLET_ADDR,
        blockchain=["eth"] 
    )
)

for asset in assets:
    # 'tokenDecimals' helps format the raw balance if needed
    print(f"Token: {asset.tokenSymbol}")
    print(f"Chain: {asset.blockchain}")
    print("-" * 20)

account = Account.from_key(PRIVATE_KEY)

# 2. Prepare input data
input_text = "Yuqi to self transaction test, 2nd times"
input_data = "0x" + input_text.encode('utf-8').hex()

# 3. Get transaction details
# Note: Ensure you are connected to the correct chain (Mainnet/Goerli/etc.)
chain_id = ankr_w3.eth.chain_id
nonce = ankr_w3.eth.get_transaction_count(account.address)
gas_price = ankr_w3.eth.gas_price

# 4. Build the transaction dictionary
tx = {
    'nonce': nonce,
    'to': MY_ANOTHER_WALLET_ADDR,
    'value': Web3.to_wei(0.01, 'ether'),          # Transfer 0.03 ETH
    'gas': 100000,       # Gas limit (sufficient for basic transfer + data)
    'gasPrice': gas_price,
    'data': input_data,
    'chainId': chain_id
}

# 5. Sign the transaction
signed_tx = account.sign_transaction(tx)

# 6. Send the transaction
# try:
#     tx_hash = ankr_w3.eth.send_raw_transaction(signed_tx.rawTransaction)
#     print(f"Transaction sent! Hash: {tx_hash.hex()}")
# except Exception as e:
#     print(f"Error sending transaction: {e}")

tx_details = ankr_w3.eth.get_transaction(TRANSACTION_20251220_HASH)

print(f"Transaction Hash ID: {TRANSACTION_20251220_HASH}")
print(f"Transaction Value (eth): {Web3.from_wei(tx_details['value'], 'ether')}")
print(f"Block Number: {tx_details['blockNumber']}")
input_data = tx_details.get('input', '0x')
print(f"Input Data: {input_data}")

block_number = tx_details['blockNumber']
block_details = ankr_w3.eth.get_block(block_number)
timestamp = block_details['timestamp']
readable_time = datetime.fromtimestamp(timestamp)
print(f"Transaction Time: {readable_time}")

print("-" * 20)

# Fetch the receipt to get gasUsed and effectiveGasPrice (receipt exists only after mining)
receipt = ankr_w3.eth.get_transaction_receipt(TRANSACTION_20251220_HASH)
if receipt is None:
    # Optionally wait for the receipt if you want to block until mined:
    try:
        receipt = ankr_w3.eth.wait_for_transaction_receipt(TRANSACTION_20251220_HASH, timeout=120)
    except Exception:
        print("Transaction not mined yet; cannot compute fee.")
        receipt = None

if receipt is not None:
    gas_used = int(receipt.get('gasUsed', receipt.get('gas_used', 0)))
    # Prefer EIP-1559 effectiveGasPrice from receipt; fall back to tx gasPrice (legacy txs)
    effective_gas_price = receipt.get('effectiveGasPrice') or tx_details.get('gasPrice') or 0

    fee_wei = gas_used * int(effective_gas_price)
    fee_eth = Web3.from_wei(fee_wei, 'ether')

    print(f"Gas used: {gas_used}")
    print(f"Effective gas price (wei): {effective_gas_price}")
    print(f"Transaction fee (wei): {fee_wei}")
    print(f"Transaction fee (ETH): {fee_eth}")
else:
    print("Skipping fee calculation because receipt is not available.")
