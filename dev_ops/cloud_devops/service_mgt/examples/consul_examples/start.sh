# Install consul (for mac)
brew install consul

# Start consul server
consul agent -dev

# Install py client
pip install python-consul

# Start two clients
python dev_ops/cloud_devops/examples/consul_examples/consul_py_client_a.py
python dev_ops/cloud_devops/examples/consul_examples/consul_py_client_b.py