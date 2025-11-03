# K8S Practice

## Common `kubectl` Debug Cmds

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

## General Standard Practice for Deployment Setup

### The deployment of The App

Assume a user developed an app `nginx` and wants to deploy it.

#### Declare the Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-deployment
  labels:
    app: my-app
```

#### `spec` for Pod Control on A Cluster Scope

`replicas: 2` tells the Deployment to create and maintain two identical copies (Pods).

`selector.matchLabels` tells what pods to manage that MUST match `spec.template.metadata.labels` by which K8S conduct further management.

```yaml
spec:
  replicas: 2
  selector:
    matchLabels:
      app: my-app
```

#### `template` by which the actual control put on the pod

The `template.spec` shows what the actual control put on the pod.

```yaml
spec.template:
  metadata:
    labels:
      app: my-app
  spec:
    ...
```

where `metadata.labels` are the labels that will be applied to each Pod created by this Deployment.
As mentioned, this `app: my-app` label matches the Deployment's `selector`, allowing the Deployment to find and manage its Pods.

The secondary `spec.template.spec` shows control on individual pods.

#### About security and execution privileges

* `securityContext.runAsNonRoot: true`: must NOT run as `root`
* `securityContext.runAsUser: 1001`: run its main process with the user ID `1001`. This ensures the container runs with minimal privileges, significantly reducing the potential damage if the application is compromised.

```yaml
spec.template.spec:
  # --- Security Best Practices ---
  securityContext:
    runAsNonRoot: true
    runAsUser: 1001
```

#### Container Control

Container Management by `containers`: A Pod can run one or more containers. This is a list of those containers (indicative of the `- name`).

* `name`: A name for the container within the Pod.
* `image`: The container image to be pulled and run (e.g., from Docker Hub or another registry).
* `ports`: This specifies the network ports the container **exposes**. 
* `livenessProbe`: This probe checks if the container is still running correctly. If the liveness probe fails, Kubernetes will **restart** the container to try and fix the problem.
* `readinessProbe`: This probe checks if the container is ready to start accepting traffic. If the probe fails, Kubernetes will **not send traffic** to the Pod, even if it's running.

##### Probe Configuration

* `httpGet`: Performs an HTTP GET request to check health.
* `initialDelaySeconds`: How long to wait after the container starts before performing the first probe.
* `periodSeconds`: How often to perform the probe.

```yaml
spec.template.spec:
  containers:
  - name: my-app
    image: "nginx:latest" # Replace with your actual app image
    ports:
    - containerPort: 80
    # --- Health Checks ---
    livenessProbe:
      httpGet:
        path: /
        port: 80
      initialDelaySeconds: 5
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /
        port: 80
      initialDelaySeconds: 5
      periodSeconds: 5
```

##### Resource Configuration

* `requests`: This specifies the **minimum** resources the container is guaranteed to get.
* `limits`: This specifies the **maximum** resources the container is allowed to use.

```yaml
spec.template.spec.containers[0].name.resources:
  requests:
      cpu: "100m"
      memory: "128Mi"
    limits:
      cpu: "200m"
      memory: "256Mi"
```

##### Env Vars

* `configMapRef`: This tells the container to take all the key-value pairs from the `ConfigMap` named `my-app-config` and expose them as environment variables.
* `secretRef`: This takes all the key-value pairs from the Secret named my-app-secret and exposes them as environment variables. This is the correct way to handle sensitive data like API keys or passwords.

```yaml
spec.template.spec.containers[0].name.envFrom:
  - configMapRef:
      name: my-app-config
  - secretRef:
      name: my-app-secret
```

### Understand Service URL Exposure

Service types such as in `spec.type`,

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-api
  namespace: my-api-env
spec:
  type: NodePort # Exposes the service on each Node's IP at a static port.
  selector:
    app: my-api
  ports:
  - port: 8080
    targetPort: 80
    nodePort: 31234
```

there exist

$$
\text{ClusterIP}\in\text{NodePort}\in\text{LoadBalancer}
$$

* `ClusterIP`: Internal only, within the cluster.
* `NodePort`:Kubernetes does everything it does for `ClusterIP` (allocating a stable internal IP) and additionally exposes the Service on a static port (the "NodePort") on the network interface of every node in the cluster. The valid port range is typically 30000-32767. Traffic arriving at `<Node_IP>:<NodePort>` is then routed to the internal `ClusterIP`, which in turn routes it to the correct pods.
* `LoadBalancer`: Kubernetes does everything it does for `NodePort` and **additionally** makes an API call to the cloud provider (minikube, aliCloud, etc), requesting that a network load balancer be provisioned. The cloud provider creates the load balancer, gives it a stable public IP, and configures it to forward traffic to the `NodePort` on all the nodes in your cluster.

#### Best Practice to Expose URL: by Ingress

`Ingress` is used to expose multiple services through a single entry point, using user-friendly URLs with hostnames and paths.

An Ingress object acts as a smart router or reverse proxy (Layer 7). It routes external HTTP/S traffic to different services based on rules.

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-api-ingress
  namespace: my-api-env
spec:
  rules:
  - host: "api.my-company.com" # The public domain name.
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-api # Routes traffic to the 'my-api' service...
            port:
              number: 8080 # ...on port 8080.
```

#### Production URL Preparation with Ingress

Given

```yaml
rules:
  - host: "api.my-company.com" # The public domain name.
```

First need to register a domain name such as `my-company.com` (e.g., AWS Route 53).
The sub-domain does NOT incur additional domain name register fee.

In cloud provider register a public IP, e.g., `203.0.113.55`.
Then create `api.my-company.com` to bind `203.0.113.55`.

Finally, having declared `spec.rules[0].http.paths[0].path: /` and  `spec.rules[0].http.paths[0].path.backend.service.name: my-api`, this request on hitting all path with the prefix `/` will get routed to service `my-api`.

```txt
curl --header "Host: api.my-company.com" http://api.my-company.com/*
```

#### Certificate Management

The added `spec.tls` indicates enabled TLS.
`secretName: my-app-tls-secret` loads `Secret` named `my-app-tls-secret` key and crt.

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app-ingress
  namespace: my-app-namespace
spec:
  tls:
  - hosts:
      - api.my-company.com
    secretName: my-app-tls-secret
  rules:
  - host: "api.my-company.com" # This MUST match a host in the tls section.
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-app # The destination service.
            port:
              number: 80
```

#####  Set Up TLS as Secret

1. Create tls key and crt files for `api.my-company.com`, and encode them as base64

```bash
openssl req -x509 -newkey rsa:2048 \
  -keyout tls.key \
  -out tls.crt \
  -sha256 -days 365 -nodes \
  -subj "/CN=api.my-company.com"

# Encode the private key and certificate
cat tls.crt | base64 -w 0
cat tls.key | base64 -w 0
```

2. Copy the base64 files to Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: my-app-tls-secret
  namespace: my-app-namespace
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSURaekNDQWsrZ0F3SUJBZ0lSQUp5V1VwL0tCNWVtYWVJN1BFRjRKa0F3RFFZSktvWklodmNOQVFFTEJRQXcKRFRFTE1Ba0dBMVVFQmhNQ1ZWTXhFREFPQmdOVkJBTVM…..(many more lines)…..xEL50k0ge0DRyUQp84tFt2UlUgg=Ci0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K
  tls.key: LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tCk1JSUpRd0lCQURBTkJna3Foa2lHOXcwQkFRRUZBQVNDQ1Mwd2dna3lBZ0VBQW9JQkFRRGdYczRyTmF0R2s5ZApCc2ZqWFJ0d2xPZ3Y2VEF0UUVxbmhxeTVycCsxMVh…..(many more lines)…..gYjZldGs2dDBkMEdKM0J6b2RzZwpRMjA2Z3JJdG9QWXA4RzRSCi0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS0K
```

3. From ingress include the secret under `spec.tls`


```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app-ingress
  namespace: my-app-namespace
spec:
  tls:
  - hosts:
      - api.my-company.com
    secretName: my-app-tls-secret
```

### Assistant Configs

There are a few dimensions to take into consideration when launching an app:

* `ConfigMap`: Stores non-sensitive configuration data, like environment settings or feature flags.
* `HorizontalPodAutoscaler`: Automatically scales the number of pods in your deployment up or down based on CPU utilization or other metrics.
* `PodDisruptionBudget`: Protects application from voluntary disruptions, like a node being drained for maintenance.

## Disk Choices

|Volume Type|Data Persistence|Storage Location|Common Use Case|
|:---|:---|:---|:---|
|`emptyDir`|Ephemeral (tied to Pod lifecycle)|Node's local disk|Sharing data between containers; scratch space
|`hostPath`|Persistent (on the specific host)|A specific directory on the Node|Node-level agents (log collectors, monitors)
|`PV`/`PVC`|Fully Persistent (independent of Pod)|Network Storage (EBS, NFS, etc.)|Databases, user data, stateful applications|
|`configMap`/`secret`|Persistent (as K8s object)|Kubernetes API (etcd), mounted in-memory|Injecting configuration files, passwords, API keys, certificates|

where

* `emptyDir` equivalent of local disk, i.e., APP -> CPU -> Memory Bus -> Disk Controller -> Local Disk (SSD/HDD)
* `PV`/`PVC` equivalent of network disk (x1000 times slower than local disk), i.e., APP -> CPU -> Node's Network Card -> Physical Network Switches -> Storage Array's Network Card -> Storage Array's CPU/Cache -> Storage Array's Disk
* `hostPath`

## K8S Design Concepts

### Sidecar

In K8S, a Sidecar is a container that runs alongside the main application container within the same Pod providing assistant functions; they are created, started, stopped, and scaled together.

For example, a logging container is created alongside the app container.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app-with-sidecar
spec:
  containers:
  - name: main-application
    image: my-app:1.0
    ports:
    - containerPort: 8080
  - name: logging-sidecar
    image: fluentd:latest
```

### Headless Service vs Load Balancing

A headless service is a special type of Service that does NOT have a stable IP address NOR perform load balancing.
Instead of providing a single virtual IP to route traffic, a headless service directly exposes the IP addresses of the individual pods it manages through DNS records.

Headless under the hood: K8S DNS

* When to do a DNS lookup for a regular service, Kubernetes DNS returns a single A record—the stable ClusterIP of the service. Good for Stateful/peer-to-peer2 applications.
* When to do a DNS lookup for a headless service, Kubernetes DNS returns multiple A records—one for each pod that is part of the service. This response gives the client a list of all the pod IPs. Good for stateless applications.

#### Headless Service Purpose and Behavior

* Purpose: To provide discovery. Its sole job is to create DNS records that point directly to the IP addresses of each backing pod.
* Behavior: A DNS query for the service name returns **a list of** A records, one for each pod's IP address.
* Best for: **Stateful** applications or any system where pod identity and **direct peer-to-peer communication** are required (e.g., database clusters, `Zookeeper`, `Loki` with `Memberlist`).

#### Load Balancing Service (e.g., ClusterIP, LoadBalancer)

* Purpose: To provide abstraction. It hides the individual pods behind a single, stable virtual IP address.
* Mechanism:
  * Kubernetes gives the service a **stable** Virtual IP (ClusterIP).
  * `kube-proxy` on each node manages iptables (or IPVS) rules.
  * When to send traffic to the ClusterIP, these rules intercept it and forward it to the real IP address of one of the healthy backing pods, usually in a round-robin fashion.
  * The pods are treated as an anonymous, interchangeable pool of resources.
* Best for: **Stateless** applications where any instance can handle any request (e.g., web servers, stateless API backends).

### K8S Service Discovery

Service discovery in K8S is about how pods are identified and referenced in other pods.

#### Label and Selector

`spec.selector.matchLabels` vs `spec.template.metadata.labels` need to match to get recognized.

```yaml
spec:
  selector:
    matchLabels:
      app: my-api
  template:
    metadata:
      labels:
        app: my-api
```

#### Service DNS

The `metadata.name` of `backend-service` determines the URL.

* In the SAME Namespace as the Service|`http://backend-service`
* In a DIFFERENT Namespace than the Service|`http://backend-service.<service-namespace>.svc.cluster.local`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: my-api
```

## Deploy DB as Pods in K8S

Generally speaking, DB setup should not be put in K8S as pods, but to use cloud native DB for performance and data safety consideration.
If to use K8S pods to host DB, here are some guides and concerns.

### Postgres Cluster as Example in K8S

Typically in K8S DB setup, there needs at least two instances one write one read.

Loaded from `$HOSTNAME`, if it is a primary, it will load the name `postgresql-0`.
If it was a replica, the hostname will be `postgresql-1`, `postgresql-2`, ...

The primary/replica is set up via the condition `if [[ $HOSTNAME =~ -0$ ]]; then`.

The default K8S db replica cluster setup does NOT have a voting/election mechanism that on failed DB instance, there will be no master/primary DB re-select.
K8S on detecting failed DB pod, will launch a new pod attached to the same persistence volume.

```yaml
# postgres-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
data:
  postgresql.conf: |
    archive_mode = on
    archive_command = 'exit 0'
    wal_level = replica
    hot_standby = on
    max_wal_senders = 10
    max_replication_slots = 10
  pg_hba.conf: |
    host replication replicator all md5
    host all all all md5
  init-script.sh: |
    #!/bin/bash
    set -e

    # Identify primary and replica
    if [[ $HOSTNAME =~ -0$ ]]; then
      # This is the primary
      until pg_isready -h localhost -p 5432 -U postgres; do
        echo "Waiting for primary to be ready..."
        sleep 2
      done

      # Create the replication user
      psql -U postgres -c "CREATE USER replicator WITH REPLICATION LOGIN PASSWORD '$REPLICATION_PASSWORD';"
    else
      # This is a replica
      until pg_isready -h postgres-0.postgres-headless -p 5432 -U postgres; do
        echo "Waiting for primary (postgres-0) to be ready..."
        sleep 2
      done

      # Clean up old data and perform a base backup
      rm -rf /var/lib/postgresql/data/*
      pg_basebackup -h postgres-0.postgres-headless -D /var/lib/postgresql/data -U replicator -vP -w
      
      # Create standby.signal file to indicate this is a replica
      touch /var/lib/postgresql/data/standby.signal

      # Configure the replica to connect to the primary
      echo "standby_mode = 'on'" >> /var/lib/postgresql/data/postgresql.auto.conf
      echo "primary_conninfo = 'host=postgres-0.postgres-headless port=5432 user=replicator password=$REPLICATION_PASSWORD'" >> /var/lib/postgresql/data/postgresql.auto.conf
    fi
```

To enable re-selection, for example for postgresql DB, need to run `patroni` as a sidecar besides each DB pod (the above example not shown though).

To access the above DB setup, set up the services.

```yaml
# postgres-services.yaml

# Service for WRITE operations
apiVersion: v1
kind: Service
metadata:
  name: postgres-primary
spec:
  ports:
  - port: 5432
  selector:
    app: postgres
    # This selector is VERY specific: it ONLY targets the pod named postgres-0
    statefulset.kubernetes.io/pod-name: postgres-0
---
# Service for READ operations
apiVersion: v1
kind: Service
metadata:
  name: postgres-replica
spec:
  ports:
  - port: 5432
  selector:
    app: postgres
    # This selector is VERY specific: it ONLY targets the pod named postgres-1
    statefulset.kubernetes.io/pod-name: postgres-1
```

For applications to access to write/read the DB, first set up secrets.
Then declare the `my-app` retrieve `DB_USER` and `DB_PASSWORD` from secrets.

Inside the `my-app` container, `WRITE_DATABASE_URL` and `READ_DATABASE_URL` are declared env vars.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-db-credentials # The name application will use to reference this secret
type: Opaque # The default type for arbitrary key-value pairs
data:
  # Key for the username is 'username'. The value is the base64 encoded 'app_user'
  username: YXBwX3VzZXI=
  # Key for the password is 'password'. The value is the base64 encoded 'secret_pass'
  password: c2VjcmV0X3Bhc3M=
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  # ... other deployment specs
  template:
    spec:
      containers:
      - name: my-app-container
        image: my-app-image
        env:
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: app-db-credentials # Must match the secret's metadata.name
              key: username           # Must match the key in the secret's data
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: app-db-credentials # Must match the secret's metadata.name
              key: password           # Must match the key in the secret's data
        - name: WRITE_DATABASE_URL
          value: "postgresql://$(DB_USER):$(DB_PASSWORD)@postgres-primary:5432/app_db"
        - name: READ_DATABASE_URL
          value: "postgresql://$(DB_USER):$(DB_PASSWORD)@postgres-replica:5432/app_db"
```

### Persistent Volume

### DB Monitoring

Typically, there are three aspects:

* the pod is alive
* internally the DB is healthy
* traffic is normal

|Monitoring Method|Perspective|Question Answered|Failure it Detects|Tool Used|
|:---|:---|:---|:---|:---|
|K8s Probes|Inside the Pod (Kubelet's view)|Pod is alive ?|A crashed process, a database that is still starting up.|`livenessProbe`, `readinessProbe`|
|White-Box|Inside the Database|Workload is heavy ?|High CPU, low cache hit ratio, slow queries, disk filling up.|`postgres-exporter`|
|Black-Box|Outside the Pod (Client's view)|"Are you reachable and available over the network?"|A NetworkPolicy blocking traffic, a CNI routing issue.|`blackbox-exporter`|

## Popular Assistant K8S Tools

### Blackbox Exporter

It probes application endpoints from an external perspective to assess their health and performance.
This is in contrast to "white-box monitoring," where monitor functions have access to the internal metrics of an application.

#### Why Need a Blackbox

Prometheus's Job (Direct Scrape): Prometheus's fundamental job is to make an HTTP GET request to an endpoint (like `/metrics`) and parse the text body of the response, expecting it to be in the "Prometheus metrics format" (e.g, some specified json).
However, it does NOT tell what err the probed service would be.

In fact, for http://my-app health check, there should ask

* Does the DNS name resolve?
* Can a TCP connection be established?
* Does it respond with a successful HTTP status code (like 200 OK)?
* How long did the entire process take (latency)?
* Is its SSL certificate valid?

### BusyBox

BusyBox is a lightweight and versatile container image that packages a collection of common Unix utilities into a single, small executable,
providing essential tools like `sh`, `ls`, `wget`, `nslookup`, and `ping` in a minimal footprint。


