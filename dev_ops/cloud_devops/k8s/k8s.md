# Kubernetes

## Concepts

* Container: an app image, e.g., docker container
* Pod: a "logic host", $\text{Container} \in \text{Pod}$
* Node: a worker virtual/physical machine, $\text{Pod} \in \text{Node}$
* Control Plane coordinates the cluster, such as scheduling the containers to run on the cluster's nodes.
* Cluster: many virtual/physical machines, $\text{Node} \in \text{Cluster}$

### Tools

* `kubelet`: a node-level agent that is in charge of executing pod requirements, managing resources, and guaranteeing cluster health; each node has a Kubelet, which is an agent for managing the node and communicating with the Kubernetes control plane. 
* `kubeadm`: a tool that is used to bootstrap a Kubernetes cluster from scratch
* `kubectl`: a command line tool that you use to communicate with the Kubernetes API server.

### K8S Internal Management

* api-server: Exposes the Kubernetes API
* etcd: Key-value store for cluster data
* scheduler: Watches for newly created pods and assigns a worker node to run them.
* controller-manager: Runs a control loop that watches the state of the cluster through the api-server and if necessary, moves the current state to the desired state.

### Common `kubectl` Use Cases and Debug

For example, here to find what host addr and port `grafana` is running.

1. Check the pod

```sh
kubectl get pods -n <namespace> -l app.kubernetes.io/name=grafana -o wide
```

Is it Ready (e.g., 1/1 or 2/2)?

2. Check the service

Is TYPE NodePort? Must got a node port so that it is accessible externally.
Does PORT(S) show 80:30080/TCP? (Internal port 80 mapped to external NodePort 30080).

```sh
kubectl get svc -n <namespace> -l app.kubernetes.io/name=grafana
```

Assume the service returns this

```txt
NAME                      TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
prom-stack-grafana   NodePort   10.96.246.174   <none>        80:30080/TCP   142m
```

3. Further check the service

```sh
kubectl describe svc prom-stack-grafana -n <namespace>
```

Port and NodePort: should match 80 and 30080.
TargetPort: should be the port Grafana listens on inside the pod.
Endpoints: should list the IP address and port of your running Grafana pod(s).

Assume

```txt

Name:                     prom-stack-grafana
Namespace:                monitoring
...
Type:                     NodePort
IP Family Policy:         SingleStack
IP Families:              IPv4
IP:                       10.96.246.174
IPs:                      10.96.246.174
Port:                     http-web  80/TCP
TargetPort:               3000/TCP
NodePort:                 http-web  30080/TCP
Endpoints:                10.244.0.54:3000
Session Affinity:         None
External Traffic Policy:  Cluster
Events:                   <none>
```

## Pod and Deployment

A Pod is a Kubernetes representation of a functioning "logic host", included of one or many containers and shared resources.
Pods are the smallest deployable unit in Kubernetes.

Some shared resources for those containers. Those resources include:

1. Shared storage, as Volumes
2. Networking, as a unique cluster IP address
3. Information about how to run each container, such as the container image version or  specific ports to use

<div style="display: flex; justify-content: center;">
      <img src="imgs/pod.png" width="25%" height="25%" alt="pod" />
</div>
</br>

One pod can have multiple containers.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: test-container
    image: registry.k8s.io/busybox
    env:
    - name: DB_URL
      value: postgres://db_url:5432
  - name: proxy-container
    image: envoyproxy/envoy:v1.12.2
    ports:
      - containerPort: 80
```

where `busybox` coming in somewhere between 1 and 5 Mb in on-disk size (depending on the variant), it combines tiny versions of many common UNIX utilities into a single small executable as an compressed version of such tools.

### `kind: Pod` vs `kind: Deployment`

Pods are the smallest deployable unit in Kubernetes, and there is no much additional config that can do about `kind: pod`. Here `kind: Deployment` comes in to rescue as preferred with added configs such as relaunched dead pod to maintain minimal replicas and logging/monitoring.

check a service by `kubectl describe service hello-world`.

In `Deployment`, K8s creates two different objects: a Pod definition, using as its specification what is available in the "template" field of the Deployment, and a ReplicaSet.

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello-world
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hello-world 
  template:
    metadata:
      labels:
        app: hello-world
    spec:
      containers:
      - name: hello-world-pod
        image: webapp-flask-helloworld-docker:latest
        imagePullPolicy: Never
        ports: 
        - containerPort: 80
        - containerPort: 443
```

### `template`

`template` describes how to config pods in `Deployment`, `Job`, etc.

### `Job` vs `Deployment`

The main difference between `Deployment`s and `Job`s is how they handle a Pod that is terminated. A Deployment is intended to be a "service", e.g. it should be up-and-running, so it will try to restart the Pods it manage, to match the desired number of replicas. While a Job is intended to execute and successfully terminate.

Jobs ensure that a specified number of pods complete successfully, whereas Deployments maintain a desired number of replicas, continuously monitoring and managing their state.


### Deployment Strategy

A K8S Deployment strategy encompasses the methods of creating, upgrading, or downgrading to a different version of a K8S application (typically a pod/container).

* Recreate
Demise old pods then launch new pods. There will be downtime during this strategy deployment

* RollingUpdate
When new launched pods are in service that available pods exceed the minimal, then start demising old pods. Recycling are through updating the pods according to the parameters: `maxSurge` and `maxUnavailable`.

`.spec.strategy.rollingUpdate.maxUnavailable` is an optional field that specifies the maximum number of Pods that can be unavailable during the update process.

`.spec.strategy.rollingUpdate.maxSurge` is an optional field that specifies the maximum number of Pods that can be created over the desired number of Pods.

For example, to recycle pod one by one, use the config below.

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
     maxSurge: 1
     maxUnavailable: 0 
```

### StatelessSet vs StatefulSet

A stateless application is one that does not care which network it is using, and it does not need permanent storage. Examples of stateless apps may include web servers (Apache, Nginx, or Tomcat).

On the other hand, a stateful applications have persistent/modifiable data, such as DBs.

## Labels and Selector

`labels` and `selector` are used to group and reference resources.

Popular labels are
`"release" : "stable"`, `"release" : "canary"`,
`"environment" : "dev"`, `"environment" : "qa"`, `"environment" : "production"`.

A popular use case is that, in `apiVersion: apps/v1` implementation of `kind: Deployment`, `labels.app.hello-world` should match selector's `matchLabels` so that this deployment can bind the app's resource.

```yaml
...
labels:
    app: hello-world
...
spec:
  selector:
    matchLabels:
      app: hello-world 
```

In `Service`, networking is bound to corresponding `Deployment` via `selector`, and `type: LoadBalancer` sets data routing from the load balancer to the bound pods.

```yaml
selector:
  app: hello-world
...
type: LoadBalancer
```

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

## Services

Service is a method for exposing a network application that is running as one or more Pods in cluster.

### Service Port Type 

Reference: https://kubernetes.io/docs/concepts/services-networking/service/

The available type values and their behaviors are:

* ClusterIP
Exposes the Service on a cluster-internal IP. Choosing this value makes the Service only reachable from within the cluster. This is the default that is used if you don't explicitly specify a type for a Service. You can expose the Service to the public internet using an Ingress or a Gateway.

* NodePort
Exposes the Service on each Node's IP at a static port (the NodePort). To make the node port available, Kubernetes sets up a cluster IP address, the same as if you had requested a Service of type: ClusterIP.

* LoadBalancer
Exposes the Service externally using an external load balancer. Kubernetes does not directly offer a load balancing component; you must provide one, or you can integrate your Kubernetes cluster with a cloud provider.

* ExternalName
Maps the Service to the contents of the externalName field (for example, to the hostname api.foo.bar.example). The mapping configures your cluster's DNS server to return a CNAME record with that external hostname value. No proxying of any kind is set up.

To implement a Service of `type: LoadBalancer`, Kubernetes typically starts off by making the changes that are equivalent to you requesting a Service of type: `NodePort`. The cloud-controller-manager component then configures the external load balancer to forward traffic to that assigned node port.

`containerPort: 80`: Indicates that the container will listen on port 80.
For example, a mysql container has `containerPort: 3306`.

There can be multiple container port listening, such as

```yaml
ports: 
    - containerPort: 80
    - containerPort: 443
```

## Ingress

`Ingress` is an API object that manages external access to the services in a cluster, typically HTTP.

It provides functions such as firewall, load balancing, virtual DNS name hosting, etc.

<div style="display: flex; justify-content: center;">
      <img src="imgs/ingress.svg" width="40%" height="25%" alt="ingress.svg" />
</div>
</br>

### Ingress Controller and Ingress Class

In order for the Ingress resource to work, the cluster must have an *ingress controller* running.

Ingresses can be implemented by different controllers with different configs
An `IngressClass` is the controller implementation that contains additional configuration including the name of the controller.

Default `IngressClass` is used when setting `ingressclass.kubernetes.io/is-default-class: true` or `ingressClassName` is not set.

Below is an example that implements nginx as the controller.

```yaml
apiVersion: networking.k8s.io/v1
kind: IngressClass
metadata:
  labels:
    app.kubernetes.io/component: controller
  name: nginx-example
  annotations:
    ingressclass.kubernetes.io/is-default-class: "true"
spec:
  controller: k8s.io/ingress-nginx
```

### Specifications

* Access Rules

|Path Type `pathType`|Rule|Example Path|Match|
|-|-|-|-|
|Prefix|\ |(all paths) |Yes|
|Exact| \foo|\foo |Yes|
|Exact| \foo|\foo\ |No|
|Prefix| \foo|\foo |Yes|
|Prefix| \foo|\foo\ |Yes|
|Prefix| \foo|\foo\bar |Yes|

* Backend

`backend` describes what services are assigned to handles requests.

```yaml
...
- http:
      paths:
      - pathType: Prefix
        path: "/"
        backend:
          service:
            name: hello-world-service
            port:
              number: 80
```

* Hostname

Ingress can provide DNS name hosting.

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ingress-wildcard-host
spec:
  rules:
  - host: "foo.bar.com"
    http:
        ...
  - host: "*.foo.com"
    http:
        ...
```

To get k8s locally host an DNS name;
reference: https://minikube.sigs.k8s.io/docs/handbook/addons/ingress-dns/

## Storage `PersistentVolume` (PV) and `PersistentVolumeClaim` (PVC)

A `PersistentVolume` (PV) is a piece of storage in the cluster provisioned by an administrator or dynamically allocated using Storage Classes, and it has a lifecycle independent of any individual Pod that uses the PV.

A `PersistentVolumeClaim` (PVC) is a request for storage by a user, similar to a Pod consuming node resources, but PVC consuming PV resources.
Claims can request specific size and access modes (e.g., they can be mounted ReadWriteOnce, ReadOnlyMany).

* Host Path

A `hostPath` volume mounts a file or directory from the host node's filesystem into your Pod.

A popular use case is `hostPath.path: /var/log` for node-level system centralized logging.

* Storage Class

*Storage class* is used to administer how volume is provisioned.
The administration includes access mode (e.g., ReadWriteOnce, ReadOnlyMany), lifecycle (e.g., to or not to be deleted after PVC ends), etc.

When a PVC does not specify a `storageClassName` or `storageClassName=""`, the default StorageClass is used; when set
`storageclass.kubernetes.io/is-default-class: true` annotation in PV and PVC, the default StorageClass is used.

A `manual` storage class is a name used in PV/PVC without actually creating a StorageClass.

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: manual
provisioner: kubernetes.io/no-provisioner
```
