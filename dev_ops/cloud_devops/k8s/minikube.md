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

## Fullstack K8S Example

* Load Balancer
* Pod Rolling Update
* Auto Scaling by Usage
* Basic Firewall/Security Config
* Message Queue Buffer for Requests
* Logging and Monitoring, Shown via Dashboard
* DB Cluster (One Writer + Two Reader)