# Service Discovery

*Service discovery* registers what APIs be bound to what services on what servers/clients, as well as API documentation and registry.

In cloud where high availability is a big concern that there are multiple servers providing the same API.
Service discovery helps monitor what servers are healthy to be queried for what APIs.

* Service Discovery

Eureka provides a lightweight service registry (client self-reporting and health check), while consul is much heavier providing more functions but easy to use.

||Eureka|Consul|
|-|-|-|
|Service Discovery|REST-based API for registration and lookup|DNS-based and HTTP-based discovery|
|Health Checks|Only client-side (self-reported)|Server-side (actively probes service health)|
|Load Balancing|No built-in LB (requires Ribbon)|Built-in service-aware load balancing|

* Latency

Eureka is slow while consul is fast.

||Eureka|Consul|
|-|-|-|
|Service Registration Latency|Fast (~10ms)|Moderate (~50-100ms)|
|Service Discovery Latency|Very fast (~5ms, cached)|Slightly higher (~20-50ms, due to consensus)|
|Health Check Frequency|Clients send periodic heartbeats (~30s default)|Server actively checks services (intervals configurable)|

* CAP Theorem

Eureka is AP-focused while consul is CP-focused.

||Eureka|Consul|
|-|-|-|
|CAP Theorem Focus|AP (Availability & Partition Tolerance)|CP (Consistency & Partition Tolerance)|
|Behavior Under Network Partition|Allows stale service registrations to ensure availability, even if nodes cannot sync.|Prioritizes consistency—may reject queries if it cannot sync with a quorum of nodes.|

## Consul

* Developed by HashiCorp.
* Written in Go.
* Key-Value Store: A hierarchical key-value store for dynamic configuration.

A typical client example use is as below

1. connect to consul server

```py
# Default: Consul(host='127.0.0.1', port=8500, token=None, scheme='http', consistency='default', dc=None, verify=True, cert=None)
c = consul.Consul()
```

2. register the process itself to consul server

```py
c.agent.service.register(
    name=this_service_name,
    service_id=this_service_id,
    address=this_addr,
    port=this_port,
    check=consul.Check.http(f'http://{this_addr}:{this_port}/health', interval='10s')
)
```

3. Get a healthy `{ip_addr}:{port}` from consul to make a request

```py
while True:
    services = c.health.service(another_service_name, passing=True)[1]
    if services:
        found_another_service_address = services[0]['Service']['Address']
        found_another_service_port = services[0]['Service']['Port']
        url = f"http://{found_another_service_address}:{found_another_service_port}/hello"
        response = requests.get(url)
        print(f"Response from {another_service_name}: {response.content.decode()}")
    else:
        print(f"No healthy {another_service_name} found")
```

4. (Optional) Put/get to share cloud-wise config

```py
c.kv.put('config/feature_x_config', '{"is_enabled": true}')
time.sleep(1)
feature_x_config = json.loads(c.kv.get('config/feature_x_config')[1]['Value'].decode('utf-8'))
if feature_x_config.get("is_enabled") == True:
  print("This feature feature_x_enabled is enabled.")
```

1. Remember to deregister from consul if the service process is shutdown

```py
# Define a function to deregister the service from Consul
def deregister_service():
    c.agent.service.deregister(this_service_id)
    print(f"Service {this_service_id} deregistered from Consul")

# Register the deregistration function to run when the app shuts down
atexit.register(deregister_service)
```

### Cluster Management and Raft Consensus Voting

*Raft* is a consensus algorithm designed to manage replicated logs in distributed systems, ensuring that multiple nodes agree on a shared state.
Raft is used for a cluster of nodes to maintain consistency by replicating a log of operations across all nodes, so they execute the same sequence of commands.

* Leader-Based: A single node, called the leader, handles all client requests and coordinates log replication to other nodes.
* Log Replication: The leader appends commands to its log and ensures they are replicated to follower nodes.
* Safety: Raft guarantees that only log entries that are safely committed (replicated to a majority of nodes) are applied to the system's state.

### Consul Key/Value Store CAP Study

Given *Consistency*, *Availability*, and *Partition Tolerance* (CAP Principle),

Consul key/value store sacrificed high availability to guarantee strong consistency.

Partition Tolerance is achieved that when nodes are experiencing partition, by default old config data is provided, unless specified when partition is done.

## Eureka

* Developed by Netflix as part of the Netflix OSS (Open Source Software) stack.
* Written in Java.
* No built-in Key-Value Store

By 2024, Eureka server launch must be with java springboot.