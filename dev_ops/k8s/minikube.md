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

### 
```

```bash
minikube start --driver=docker
```