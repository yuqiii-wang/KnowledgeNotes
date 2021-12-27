from web3 import Web3

eth_provider="https://mainnet.infura.io/v3/<endpoint>"

# HTTPProvider:
w3 = Web3(Web3.HTTPProvider(eth_provider))
res = w3.isConnected()
print(res)

# print blk
latest_block = w3.eth.getBlock('latest')
print(latest_block)

balance = w3.eth.getBalance('0xaa82b089AE495d51bbC2aC106E712Fa4Ef725D97')
print(balance) # Wei

trans = w3.eth.getTransaction('0xc90378a4418d4613dcb39d7215f6e9dabe46b77db0bc0021b25faf7c570f6789')
print(trans)