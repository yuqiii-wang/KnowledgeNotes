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

## K8S Resources

Resources are K8S managing

### Workload Resources

Workload resources are used to manage and run applications on cluster.

#### Pod

The smallest and most basic deployable object in Kubernetes.
A Pod represents a single instance of a running process in cluster and can contain one or more containers. Containers within the same Pod share the same network and storage resources.

#### Deployment

Deployments provide declarative updates to Pods to easily manage replicas, perform rolling updates, and roll back to previous versions.

#### StatefulSet

Used for managing stateful applications that require stable, unique network identifiers and persistent storage.
This is ideal for applications like databases (e.g., MySQL, PostgreSQL) and message queues.

#### DaemonSet

Ensures that all (or some) Nodes run a copy of a Pod.
This is useful for cluster-level services like log collectors or monitoring agents.

#### Job

Creates one or more Pods and ensures that a specified number of them successfully terminate.Jobs are designed for tasks that run to completion, such as batch processing or backups.

#### CronJob

Manages time-based Jobs. It's used for tasks that need to be executed on a schedule, like periodic backups or report generation.

#### ReplicaSet

Ensures that a specified number of Pod replicas are running at any given time.
Deployments are a higher-level resource that manage ReplicaSets, and it's generally recommended to use Deployments directly.

### Service Discovery and Networking Resources

These resources are responsible for exposing your applications and managing network traffic between them.

#### Service

An abstraction that defines a logical set of Pods and a policy by which to access them.
Services provide a stable IP address and DNS name, enabling communication between different parts of an application.Kubernetes offers different types of services, including `ClusterIP`, `NodePort`, and `LoadBalancer`.

#### Ingress

Manages e**xternal access to services** within a cluster, typically HTTP and HTTPS.
Ingress can provide load balancing, SSL termination, and name-based virtual hosting.

#### EndpointSlice

A resource that represents a subset of the network endpoints of a Service. It allows for more scalable and efficient service discovery.

### Storage Resources

These resources manage how data is stored and accessed by applications.

#### PersistentVolume (PV)

A piece of storage in the cluster that has been provisioned by an administrator or dynamically provisioned using `StorageClasses`.

PVs are independent of the Pod lifecycle, ensuring that data persists even if Pods are recreated.

#### PersistentVolumeClaim (PVC)

A request for storage by a user. It's similar to how a Pod consumes Node resources, a PVC consumes PV resources.

#### StorageClass

Provides a way for administrators to describe the "classes" of storage they offer.
Different classes might map to quality-of-service levels, backup policies, or arbitrary policies determined by the cluster administrators.This allows for dynamic provisioning of storage.

#### Volume

A directory containing data, which is accessible to the containers in a Pod. Kubernetes supports various types of volumes to provide storage to containers.

### Configuration and Secret Management

These resources help manage application configuration and sensitive data separately from application code.

#### ConfigMap

Used to store non-confidential configuration data in key-value pairs.
This allows you to decouple configuration from your application logic.

#### Secret

Used to store and manage sensitive information such as passwords, OAuth tokens, and ssh keys.
Storing this information in a Secret is more secure than putting it verbatim in a Pod definition or a container image.

### Cluster-Level Resources
These resources are used to manage the overall behavior and organization of the cluster.

#### Namespace

A way to divide cluster resources between multiple users or teams.
It provides a scope for names, allowing you to have resources with the same name in different namespaces.

#### Role and ClusterRole

Define permissions for resources. A Role grants permissions within a particular namespace, while a ClusterRole grants permissions cluster-wide.

#### RoleBinding and ClusterRoleBinding

Grant the permissions defined in a Role or ClusterRole to a set of users.
ServiceAccount: Provides an identity for processes that run in a Pod.

## K8S Config Yaml Syntax

### The Four Top-Level Fields

#### `apiVersion`

The specific `apiVersion` determines the available features and the structure of the spec field for the object user are creating.

#### `kind`

Some of the most common kinds include `Pod`, `Deployment`, `Service`, and `Namespace`.

#### `metadata`

It helps uniquely identify the object within the Kubernetes cluster.

Typically, there are

* `name`: A unique name for the object within its namespace.
* `namespace`: An optional field that specifies the virtual cluster to which the object belongs.
* `labels`: tags helpful to selection. For example, you might label all the components of a specific application with `app: my-webapp`.

#### `spec`

The specifications define the details of resource.

The structure and content of the `spec` section are highly dependent on what `kind` the resource is.

* `template` in `spec`

This section defines the blueprint for the Pods that the Deployment will create.
It has its own metadata (with labels) and spec for the containers within the Pods.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app-container
        image: nginx:1.14.2
        ports:
        - containerPort: 80
```

### `volumes` vs `volumeMounts`

In short, `volumes` is the declaration of the use of file, while `volumeMounts` gives the actual file name to read.

Consider a use case to deploy an nginx server with written `/etc/nginx/conf.d/default.conf` and `/usr/share/nginx/html/index.html` as the index homepage.
Assume that a config map `webserver-content` is already finished and stored in `etcd`.

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  # The name of our ConfigMap, which the volume will reference.
  name: webserver-content
  namespace: default
data:
  # The first key-value pair. The key 'custom.conf' will be used as a filename.
  custom.conf: |
    server {
        listen 80;
        server_name localhost;

        root /usr/share/nginx/html;
        index index.html;

        location / {
            try_files $uri $uri/ =404;
        }
    }2
```

Below sees two separate `volumeMounts` but they both reference the same `volume` (`nginx-files-volume`).

Instead of mounting the entire volume (which would create a directory with two files), `subPath` tells Kubernetes to only take the value associated with the `custom.conf key` from ConfigMap and mount it as the file `default.conf` (also apply for `index.html` in the second file).

Nginx automatically loads `.conf` files from this directory.

```yaml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-webserver
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      volumes:
      - name: nginx-files-volume         # 1. We declare a volume and give it a logical name for this Pod.
        configMap:                       # 2. We specify its source is a ConfigMap.
          name: webserver-content        # 3. We tell it to use the ConfigMap named 'webserver-content'.

      containers:
      - name: nginx-container
        image: nginx:latest
        ports:
        - containerPort: 80
        volumeMounts:
        - name: nginx-files-volume                   # 4a. Use the volume we declared above.
          mountPath: /etc/nginx/conf.d/default.conf  # 5a. Mount it at this exact file path inside the container.
          subPath: custom.conf                       # 6a. IMPORTANT: Only use the 'custom.conf' key from the ConfigMap.

        - name: nginx-files-volume                    # 4b. Use the SAME volume again.
          mountPath: /usr/share/nginx/html/index.html # 5b. Mount it at the default homepage path.
          subPath: index.html                         # 6b. IMPORTANT: This time, only use the 'index.html' key.
```

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

The Control Plane (master node) is responsible for managing the entire cluster. It maintains the desired state of the cluster, handles scheduling of applications, and responds to events like a pod or node failure.

* API Server: acts as Cluster Gateway, functions such as authentication, routing requests to diff processes
* Scheduler: receives request from API Server and schedule launching pods based on rules (such as CPU and memory usage)
* Controller Manager: detects/monitors cluster, and launches new pods when pods fail
* etcd: has key/value store that tells pods' and nodes' information, such as health of running pods

### Worker Node

A Worker Node is the machine where applications actually run. It receives instructions from the Control Plane and manages the networking and execution of containers.

* Kubelet: the primary "node agent" that runs on each node. It takes a set of PodSpecs (yaml or json config files) and make sure a pod run accordingly to the PodSpecs.
* Kube Proxy (`kube-proxy`): Kube Proxy's job is to manage the network rules on the node that allow for communication to and from pods. It ensures a request be sent to a Service's IP, that request is intelligently forwarded to one of the correct backing Pods, even if that Pod is on a different node. It does this by directly manipulating the node's networking rules (using tools like iptables or IPVS).
* Container Runtime: `containerd` or `docker`

### Pods run on nodes

Each node has a Kubelet, which is an agent for managing the node and communicating with the Kubernetes control plane.

<div style="display: flex; justify-content: center;">
      <img src="imgs/node.png" width="40%" height="40%" alt="node" />
</div>
</br>

## Kubernetes Cluster's Database (`etcd`)

`etcd` is a highly available, consistent, distributed key-value store, the single source of truth for all cluster data.

`etcd` stores the desired state and the current state of every object in the Kubernetes cluster. This includes:

* Workload Definitions: The complete YAML/JSON definitions for every `Pod`, `Deployment`, `StatefulSet`, `Service`, etc.
* Cluster State: The real-time status of those objects. For example: "Pod grafana-xyz is currently Running on worker-node-2 with IP 10.42.2.5."
* Configuration and Secrets: All ConfigMap and Secret objects.
* Network Information: `Service` definitions, `Ingress` rules, and network policies.
* RBAC Rules: All roles, cluster roles, and their bindings that define who can do what in the cluster.

A typical use in config map is that (as an example):

Write config to a file `alertmanager.yml`.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config  #<-- The source name
data:
  alertmanager.yml: |-     #<-- This key becomes the filename
    route:
      receiver: 'do-nothing'
    receivers:
    - name: 'do-nothing'
```

Deploy the config map via matching `configMap.name: alertmanager-config`.
The `'args'` array is part of the container's definition

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
spec:
  template:
    spec:
      volumes:
      - name: config-volume            # <-- Nickname is defined here
        configMap:
          name: alertmanager-config  # <-- Points to the ConfigMap's real name
      containers:
      - name: alertmanager
        image: prom/alertmanager:latest
        args: # <-- The 'args' array is part of the container's definition
          - "--config.file=/etc/alertmanager/alertmanager.yml"
        volumeMounts:
        - name: config-volume            # <-- Must match the nickname from Part 1
          mountPath: /etc/alertmanager/  # <-- The destination directory inside the container
```
