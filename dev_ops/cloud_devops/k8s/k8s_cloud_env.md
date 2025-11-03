# K8S Cloud Env

## Cloud Provided Env

K8S is useful in app and networking orchestration with assistant tools to increase resiliency, cyber security, monitoring/visibility, etc.
However, in best practice below works are not delegated in purely K8S env, but with cloud provider (e.g., AliCloud, AWS, GoogleCloud, Azure) provisioning.

* For file storage: AWS S3, AliCloud SSO, not need to put files in Persistent Volume (PV)
* No need of node-level monitoring for cloud provider has already installed node monitoring tools by default, e.g., cpu and memory usage, disk volume usage
* Usually cloud provider has default log collection tools that automatically collect from `stdout` from pods, hence no need to store logs in PV
* Usually cloud provider provisioned DB has optimizations; do NOT need to use DB as container

## Local Study Env: Minikube

`minikube` is study-version Kubernetes (running on one node) env.

Start by the below.

```bash
minikube start --driver=docker
```

### Useful tips

* Use local image

Load local built image by `minikube image load <tag>:<version>`

By default, K8S pulls image online. To force K8S use local built image, should add `imagePullPolicy: Never`.

* Use `minikube dashboard` for GUI

* Mirror Site

First time run `minikube start` (first start may download many dependencies, e.g., `kubectl`).
For internet issues, there might be slow in downloading, user can manually download some most popular dependencies

```bash
sudo apt-get install -y kubelet kubeadm kubectl
```

Reference: https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/

K8S CN Mirrors:
https://kubernetes.feisky.xyz/appendix/mirrors

By the year 2024, popular k8s mirror providers in China are Azure and Aliyun.

```bash
minikube start --driver=docker --image-repository=registry.cn-hangzhou.aliyuncs.com/google_containers
```

or

```bash
minikube start --driver=docker --image-repository=http://mirror.azure.cn/kubernetes/packages/apt/
```

* API Server Start

`minikube start` automatically starts the API server.
If it fails, user might get error `https://localhost:<random_port>` not reachable/timeout.

The API server exposes an HTTP API (by default `https://localhost:<random_port>`) that lets end users, different parts of cluster, and external components communicate with one another.
Queries such as `kubectl get nodes` and `kubectl apply -f manifest.yaml` are passed to this API to interact with K8S cluster.

#### Fullstack K8S Example

* Load Balancer
* Pod Rolling Update
* Auto Scaling by Usage
* Basic Firewall/Security Config
* Message Queue Buffer for Requests
* Logging and Monitoring, Shown via Dashboard
* DB Cluster (One Writer + Two Reader)

## AliCloud K8S (ACK)

AliCloud K8S (ACK) is an K8S env provider.

ACK's true power comes from its native integration with other Alibaba Cloud services, which are managed through controllers and plugins running within the cluster.

### AliCloud Env Provisioning

#### Elastic Network Interfaces (ENIs)

#### Network Attached Storage (NAS)

#### Resource Access Management (RAM)

### Billing

*  ACK Pro (Professional) Cluster Fee

Free for basic ACK.
ACK Pro provides production level availability.

* Elastic Compute Service (ECS) Fee

Compute resources. One node one ECS.

* API server CLB Instance Fee

Hourly fee for the Classic Load Balancer (CLB) instance that exposes cluster's API server.

The Kubernetes API server is the central point for all cluster management commands (like `kubectl`). To make it accessible and highly available, ACK provisions a load balancer. This fee is for keeping that load balancer instance running.

* API server CLB LCU Fee

This is a usage-based fee for the API server's load balancer.

The more you interact with cluster's API (e.g., by `kubectl`), the higher this cost might be.

* NAT Gateway Instance Fee

fixed hourly cost for running a NAT Gateway instance.

* NAT Gateway CU Fee

Usage-based fee for the NAT Gateway, measured in Compute Units (CU).

The more connections and traffic nodes generate to the internet, the more CUs are consumed.

* EIP Configuration Fee

Elastic IP (EIP) address. An EIP is a static, public IP address.

* EIP Public Network Traffic Fee

The cost for outbound data traffic that flows from EIP to the public internet.

* Container Monitoring Pro Edition Fee

The professional version of Alibaba Cloud's Managed Service for Prometheus

The cost is based on Operations Capacity Unit (OCU).

It's a billing unit that combines multiple metrics, including the number of time-series data points collected, the volume of data stored, and the amount of data scanned for queries and alerts.
