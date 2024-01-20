# Kubernetes

* Container: an app image, e.g., docker container
* Pod: a "logic host", $\text{Container} \in \text{Pod}$
* Node: a worker virtual/physical machine, $\text{Pod} \in \text{Node}$
* Control Plane coordinates the cluster, such as scheduling the containers to run on the cluster's nodes.
* Cluster: many virtual/physical machines, $\text{Node} \in \text{Cluster}$

* `kubelet`: a node-level agent that is in charge of executing pod requirements, managing resources, and guaranteeing cluster health; each node has a Kubelet, which is an agent for managing the node and communicating with the Kubernetes control plane. 
* `kubeadm`: a tool that is used to bootstrap a Kubernetes cluster from scratch
* `kubectl`: a command line tool that you use to communicate with the Kubernetes API server. 

* api-server: Exposes the Kubernetes API
* etcd: Key-value store for cluster data
* scheduler: Watches for newly created pods and assigns a worker node to run them.
* controller-manager: Runs a control loop that watches the state of the cluster through the api-server and if necessary, moves the current state to the desired state.

## Pod

A Pod is a Kubernetes representation of a functioning "logic host", included of one or many containers and shared resources

Some shared resources for those containers. Those resources include:
1. Shared storage, as Volumes
2. Networking, as a unique cluster IP address
3. Information about how to run each container, such as the container image version or  specific ports to use


<div style="display: flex; justify-content: center;">
      <img src="imgs/pod.png" width="25%" height="25%" alt="pod" />
</div>
</br>


### Pod vs Deployment

Pods are smallest deployable unit on k8s, with not changeable configurations, volumes, etc.

For better application management, deployment comes into picture which maintains the desired state (how many instances, how much compute resource application uses) of the application. 

Deployment is declared by with `kind: Deployment`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: <app-name>
```

### StatelessSet vs StatefulSet

A stateless application is one that does not care which network it is using, and it does not need permanent storage. Examples of stateless apps may include web servers (Apache, Nginx, or Tomcat).

On the other hand, a stateful applications have persistent/modifiable data, such as DBs.

## Node

Nodes are the workers that run applications

A node is a VM or a physical computer.

There are master nodes and worker nodes:

### Master Node

* API Server: acts as Cluster Gateway, functions such as authentication, routing requests to diff processes

* Scheduler: receives request from API Server and schedule launching pods based on rules (such as CPU and memory usage)

* Controller Manager: detects/monitors cluster, and launches new pods when pods fail

* etcd: has key/value store that tells pods' and nodes' information, such as health of running pods

### Worker Node

* Kubelet: the primary "node agent" that runs on each node. It takes a set of PodSpecs (yaml or json config files) and make sure a pod run accordingly to the PodSpecs.

* Kube Proxy

* Container Runtime

### Pods run on nodes

Each node has a Kubelet, which is an agent for managing the node and communicating with the Kubernetes control plane.

<div style="display: flex; justify-content: center;">
      <img src="imgs/node.png" width="40%" height="40%" alt="node" />
</div>
</br>

## Volume

A *PersistentVolume* (PV) is a piece of storage in the cluster that has been provisioned by an administrator or dynamically provisioned using Storage Classes.

A *PersistentVolumeClaim* (PVC) is a request for storage by a user. Claims can request specific size and access modes (e.g., they can be mounted ReadWriteOnce, ReadOnlyMany or ReadWriteMany, see AccessModes).

For use of persistent volume, there should be two declared resources: PV and PVC

## Services

Services help expose groups of pods over a network by adding a layer of abstraction. This abstraction provides a stable endpoint for the pods to communicate with each other.

## A Minimal Example: Image and `.yaml` File

### Build an image

Build a docker image of a simple app by `nodejs`.

```javascript
const express = require("express");
const data = require("./data.json")

const app = express();

app.get("/", (req, res) => {
    res.send("Hello, Welcome")
})

app.listen(8000, () => {
    console.log("App is running")
});
```

```Dockerfile
FROM node:18-alpine

WORKDIR /hello-world-nodejs

COPY server.js /hello-world-nodejs/
COPY package.json /hello-world-nodejs/
COPY data.json /hello-world-nodejs/

RUN npm install

CMD ["node", "server"]
```

Build and publish the docker image.
For the used container is docker, the publish needs username and password.

```bash
docker build -t hello-world-nodejs-image .
docker image tag hello-world-nodejs-image <username>/hello-world-nodejs-image
```

### Launch the K8S cluster

A minimal working k8s cluster should see two kinds of below: `Deployment` and `Service`.

Such file to launch a k8s cluster is named `manifest.yaml`.
By default, ports are only accessible within a cluster. To access the app from external, need to expose the port via `--url`.

```bash
kubectl create -f manifest.yaml
minikube service hello-world --url
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello-world-deployment
  labels:
    app: hello-world
spec:
  selector:
    matchLabels:
      app: hello-world
  replicas: 2
  template:
    metadata:
      labels:
        app: hello-world
    spec:
      containers:
      - name: hello-world
        image: <username>/hello-world-nodejs-image
        ports:
        - containerPort: 80
        resources:
          limits:
            memory: 256Mi
            cpu: "250m"
          requests:
            memory: 128Mi
            cpu: "80m"
---
apiVersion: v1
kind: Service
metadata:
  name: hello-world
spec:
  selector:
    app: hello-world
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
      nodePort: 30081   
  type: NodePort
```

The launched services can be found by `minikube service list`, that shows external access entry `http://192.168.49.2:30081`.

```txt
|----------------------|---------------------------|--------------|---------------------------|
|      NAMESPACE       |           NAME            | TARGET PORT  |            URL            |
|----------------------|---------------------------|--------------|---------------------------|
| default              | hello-world               |           80 | http://192.168.49.2:30081 |
| default              | kubernetes                | No node port |                           |
| kube-system          | kube-dns                  | No node port |                           |
| kubernetes-dashboard | dashboard-metrics-scraper | No node port |                           |
| kubernetes-dashboard | kubernetes-dashboard      | No node port |                           |
|----------------------|---------------------------|--------------|---------------------------|
```

### Common Specs

### `selector`

### `template`

### `type: NodePort` vs `type: LoadBalancer`




### `ports`

* `port` exposes the Kubernetes service on the specified port within the cluster.
* `targetPort` on which the service will send requests to, that pod/container will be listening on.
* `containerPort` 
* `nodePort` exposes a service externally to the cluster by means of the target nodes IP address and the NodePort.

User accesses app via `nodePort`

```yaml
spec:
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
      nodePort: 30081   
```