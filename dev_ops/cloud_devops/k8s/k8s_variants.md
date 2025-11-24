# K8S Variants

Depending on development purposes, there are various K8S implementations, typically, there are

* Cloud Provided K8S Env: e.g., AliCloud ACK, AWS EKS, Google GKE
* Local Study K8S Env: Minikube
* Low-Resource Consumption K8S Env: K3S

## Cloud Provided K8S Env

K8S is useful in app and networking orchestration with assistant tools to increase resiliency, cyber security, monitoring/visibility, etc.
However, in best practice below works are not delegated in purely K8S env, but with cloud provider (e.g., AliCloud, AWS, GoogleCloud, Azure) provisioning.

* For file storage: AWS S3, AliCloud Object Storage Service (OSS), not need to put files in Persistent Volume (PV)
* No need of node-level monitoring for cloud provider has already installed node monitoring tools by default, e.g., cpu and memory usage, disk volume usage
* Usually cloud provider has default log collection tools that automatically collect from `stdout` from pods, hence no need to store logs in PV
* Usually cloud provider provisioned DB has optimizations; do NOT need to use DB as container

## Local Study Env: Minikube

`minikube` is study-version Kubernetes env.

Start by the below.

```bash
minikube start --driver=docker
```

Minikube needs additional driver as hosting env.
For example,

1. `--driver=docker` tells an already running docker daemon to run a full Linux operating system as a container.
2. Minikube then gets inside that container and installs all the necessary Kubernetes components (like `kubeadm`, `kubelet`, `etcd`, etc.).
3. The end result is a Kubernetes cluster running inside a Docker container on the host machine.

### Minikube Ingress and Tunnel

## Low-Resource Consumption K8S Env: K3S

K3s is designed for simplicity and efficiency, delivered as a single, compact binary.
This binary contains everything needed to run a Kubernetes cluster, including the container runtime (`containerd`), a CNI (Container Network Interface) plugin (Flannel), an ingress controller (Traefik), and a service load balancer.

The K3s server process consolidates the Kubernetes control plane components—API server, scheduler, and controller manager—into a single, unified process.

|Feature|K3s|Minikube|
|:---|:---|:---|
|Core Architecture|Single, lightweight binary with consolidated processes.|Virtualization-based (VM or container) with a more traditional Kubernetes process layout.
|Datastore|SQLite by default for single-node; supports etcd, MySQL, PostgreSQL for HA.|etcd.|
|Container Runtime|containerd.|Configurable, typically Docker or containerd.|
|Networking (CNI)|Flannel is bundled by default.|Does not include a CNI that supports network policies by default; requires addons like Calico.|
|Storage (CSI)|Includes a local-path-provisioner by default for host-path based storage.|Provides a storage-provisioner addon; supports host mounts.|
|Minimum Memory (RAM)|512 MB|2 GB|
|Minimum CPU|1 Core / 1 vCPU|2 Cores / 2 vCPUs|
|Idle Memory Usage|Significantly Lower (e.g., ~260-500 MB)|Higher (e.g., ~530-680 MB or more)|
|Binary/Footprint Size|Very Small (single binary < 100 MB)|Larger, plus VM/container image|

K3S is a Linux binary that is supposed to run on Linux machines.
However, 

## AliCloud K8S (ACK)

AliCloud K8S (ACK) is an K8S env provider.

ACK's true power comes from its native integration with other Alibaba Cloud services, which are managed through controllers and plugins running within the cluster.

### AliCloud Container Registry

AliCloud provides container image repo service.
Usually, depending on what cloud provider a user selects, the user should upload his/her container to the cloud's image repo, that cloud provider has optimizations over downloading/devops container image to cloud's implementation K8S.

For example to upload a container image to AliCloud,

1. login
2. tagging
3. push to upload

```sh
docker login --username=<username> registry.cn-hangzhou.aliyuncs.com
docker tag my-app:1.0 registry.cn-hangzhou.aliyuncs.com/<repo-namespace>/my-app:1.0
docker push registry.cn-hangzhou.aliyuncs.com/my-project/my-app:1.0
```

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
