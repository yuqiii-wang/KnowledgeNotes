#!/bin/bash

# Define the array of configuration file names
configs=("elasticsearch_node-1.yml"
        "elasticsearch_node-2.yml"
        "elasticsearch_node-3.yml")

# Set the Elasticsearch home directory
export ES_HOME=/path/to/elasticsearch

# Set the path to the Elasticsearch configuration file
export ES_CONF=/path/to/elasticsearch/config/elasticsearch.yml

# Define the variables for each node
NETWORK_HOST="0.0.0.0"
HTTP_PORT="9200" # Ensure this is unique if nodes are on the same machine
TRANSPORT_PORT="9300" # Set the transport port for Elasticsearch
SEED_HOSTS='["node-1", "node-2", "node-3"]'
MASTER_NODES='["node-1", "node-2", "node-3"]'

# Export variables so they can be substituted in the template
export NETWORK_HOST HTTP_PORT SEED_HOSTS MASTER_NODES

# Loop over each configuration file
for config in "${configs[@]}"; do
    # Extract the node number from the file name
    NODE_NAME=$(echo $config | grep -o -E 'node\-[0-9]+')
    echo "Gonna generate ${NODE_NAME}"

    sed "s/\${NODE_NAME}/${NODE_NAME}/g; s/\${MASTER_NODES}/${MASTER_NODES}/g; \
        s/\${HTTP_PORT}/${HTTP_PORT}/g; s/\${NETWORK_HOST}/${NETWORK_HOST}/g; \
        s/\${SEED_HOSTS}/${SEED_HOSTS}/g;" \
        elasticsearch_template.yml > ${config}

    # Start Elasticsearch with the specified transport port and configuration file
    $ES_HOME/bin/elasticsearch -Etransport.port=$TRANSPORT_PORT -Epath.conf=$ES_CONF
done

# Note: This script should be run on each node individually, with the respective configuration file present on the node.
