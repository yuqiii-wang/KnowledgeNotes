# `minikube`

`minikube` is local Kubernetes (running on one node).

All you need is Docker (or similarly compatible) container or a Virtual Machine environment, and Kubernetes is a single command away: `minikube start`


## How to Start

Download and Install
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube_latest_amd64.deb
sudo dpkg -i minikube_latest_amd64.deb
```

Make sure docker is up and running

```bash
dockerd version
```

Start by the below.

```bash
minikube start --driver=docker
```

### Trouble Shooting

First time run `minikube start` (first start may download many dependencies, e.g., `kubectl`).
For internet issues, there might be slow in downloading, user can manually download some most popular dependencies

```bash
sudo apt-get install -y kubelet kubeadm kubectl
```

Reference: https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/

User can use public repositories such as below.

```bash
minikube start --driver=docker --image-repository=registry.cn-hangzhou.aliyuncs.com/google_containers
```

## Fullstack K8S Example

* Load Balancer
* Pod Rolling Update
* Auto Scaling by Usage
* Basic Firewall/Security Config
* Message Queue Buffer for Requests
* Logging and Monitoring, Shown via Dashboard
* DB Cluster (One Writer + Two Reader)