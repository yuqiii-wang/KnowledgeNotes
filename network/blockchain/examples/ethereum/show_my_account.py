from datetime import datetime

from ankr import AnkrWeb3
from ankr.types import GetAccountBalanceRequest, GetTransactionsByAddressRequest
from eth_account import Account

from _config import (
    ETH_PROVIDER_API_KEY,
    MY_ANOTHER_WALLET_ADDR,
    TRANSACTION_20211217_HASH,
    MNEMONIC_PHRASE,
    PASSPHRASE,
    PRIVATE_KEY
)

ankr_w3 = AnkrWeb3(ETH_PROVIDER_API_KEY)

########## Coins in My Another Account ##########
assets = ankr_w3.token.get_account_balance(
    request=GetAccountBalanceRequest(
        walletAddress=MY_ANOTHER_WALLET_ADDR,
        blockchain=["eth"] 
    )
)

for asset in assets:
    # 'tokenDecimals' helps format the raw balance if needed
    print(f"Token: {asset.tokenSymbol}")
    print(f"Type: {asset.tokenType}")
    print(f"Balance: {asset.balance}") 
    print(f"Chain: {asset.blockchain}")
    print("-" * 20)

########## Tokens in My Another Account ##########

# 1. Get the generator
response = ankr_w3.query.get_transactions_by_address(
    request=GetTransactionsByAddressRequest(
        address=MY_ANOTHER_WALLET_ADDR,
        blockchain=["eth"]
    )
)

print("Searching through transaction history...")

tokens_seen = set()

# 2. Iterate directly over the generator
for tx in response:
    # Note: In the Query API, the field name is usually 'token_symbol' 
    # (snake_case) rather than 'tokenSymbol' (camelCase)
    symbol = getattr(tx, 'token_symbol', 'ETH') # Default to ETH if not found
    
    if symbol not in tokens_seen:
        print(f"Found Token: {symbol}")
        tokens_seen.add(symbol)

########## One Transaction Detail ##########
print("-" * 20)

# tx_details = ankr_w3.eth.get_transaction(TRANSACTION_20211217_HASH)

# print(f"From: {tx_details['from']}")
# print(f"To: {tx_details['to']}")
# print(f"Value (Wei): {tx_details['value']}")
# print(f"Block Number: {tx_details['blockNumber']}")
# input_data = tx_details.get('input', '0x')
# print(f"Input Data: {input_data}")

# print("-" * 20)

# block_number = tx_details['blockNumber']
# block_details = ankr_w3.eth.get_block(block_number)
# timestamp = block_details['timestamp']
# readable_time = datetime.fromtimestamp(timestamp)
# print(f"Transaction Time: {readable_time}")

# print("-" * 20)

# Account.enable_unaudited_hdwallet_features()
# account = Account.from_mnemonic(MNEMONIC_PHRASE, 
#                                 passphrase=PASSPHRASE)
        
# # 3. Extract Details
# private_key = account.key.hex()
# address = account.address

# print("Mnemonic and passphrase are VALID!")
# print(f"Unlocked Address:     {address}")

# print("-" * 20)
# account = Account.from_key(PRIVATE_KEY)
# print("Private key is VALID!")
# print(f"Unlocked Address: {account.address}")
