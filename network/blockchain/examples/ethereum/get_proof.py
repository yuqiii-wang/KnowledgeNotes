from ankr import AnkrWeb3

from _config import (ETH_PROVIDER_API_KEY,
                    MY_WALLET_ADDR)

# 1. Initialize Ankr (using their public Ethereum RPC)
ankr_w3 = AnkrWeb3(ETH_PROVIDER_API_KEY)

# 2. Set your address (using Account to ensure checksum format)
target = MY_WALLET_ADDR

# 3. Request the "Proof" which contains the 4 World State fields
state = ankr_w3.eth.get_proof(target, [], 'latest')

# 4. Output
print(f"Account Proof for: {target}")
print(f"Nonce:             {state.nonce}")
print(f"Balance:           {state.balance} Wei")
print(f"CodeHash:          {state.codeHash.hex()}")
print(f"StorageHash:       {state.storageHash.hex()}")