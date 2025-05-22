# Helm

Helm is used to simplify Kubernetes deployments and package management.

There are these key components:

* Charts: Predefined Kubernetes manifests with templating (e.g., helm create my-app).
* Repositories: Shared collections of charts (e.g., Bitnami, Artifact Hub).
* Releases: Tracked instances of deployed charts.

```txt
my-chart/  
├── Chart.yaml          # Metadata (name, version, dependencies)  
├── values.yaml         # Default configuration values  
├── templates/          # Kubernetes manifests with templating  
│   ├── deployment.yaml  
│   ├── service.yaml  
│   └── ...  
└── charts/             # Subcharts/dependencies
```

## Practices in Helm Installing Resources

Helm abstracts Kubernetes manifests into reusable "charts" (pre-configured application packages), then `kubectl apply` them.

```sh
helm install <new-unique-name> <chart> -n <namespace>
```

For example, to install `timescaledb` in k8s:

1. Get k8s cluster up and running for `helm install` will directly run pods
2. Create a namespace
3. Download `timescaledb` chart by `helm repo add`
4. Prepare `timescaledb-values.yaml` that lists custom config, e.g., host addr and port
5. Launch pods in k8s by `helm install`

```sh
minikube start
kubectl create namespace timescaledb

helm repo add timescaledb https://charts.timescale.com/
helm repo update
helm install tsdb-cluster timescaledb/timescaledb-multinode \
  -f ./timescaledb-values.yaml \
  -n timescaledb
```

`kubectl get pods -n timescaledb` shows all pods under the namespace `timescaledb` that are defined in `timescaledb-values.yaml` with one `helm install` run.

```txt
NAME                                    READY   STATUS                  RESTARTS   AGE
attachdn-tsdb-cluster-db0-data0-t4pbd   1/1     Running                 0          136m
attachdn-tsdb-cluster-db0-data1-mr2p4   1/1     Running                 0          136m
attachdn-tsdb-cluster-db0-data2-fzn65   1/1     Running                 0          136m
attachdn-tsdb-cluster-db1-data0-t98h4   1/1     Running                 0          136m
attachdn-tsdb-cluster-db1-data1-zvh2d   1/1     Running                 0          136m
attachdn-tsdb-cluster-db1-data2-c6vnp   1/1     Running                 0          136m
createdb-tsdb-cluster-db0-np2zr         1/1     Running                 0          136m
createdb-tsdb-cluster-db1-k2xs8         1/1     Running                 0          136m
tsdb-cluster-timescaledb-access-0       0/1     Init:ImagePullBackOff   0          136m
tsdb-cluster-timescaledb-data-0         0/1     Init:ImagePullBackOff   0          136m
tsdb-cluster-timescaledb-data-1         0/1     Init:ImagePullBackOff   0          136m
tsdb-cluster-timescaledb-data-2         0/1     Pending                 0          136m
```

When there is changes in `values.yaml`, run upgrade such that

```sh
helm upgrade <RELEASE_NAME> <CHART> -f values.yaml
```

### Helm vs K8S

`helm install` deploys a Helm chart (a packaged application) as a release (a named instance of the chart).
It is similar to `kubectl apply` but that works on individual YAML files (no built-in versioning or dependency management).

* Processes the chart's templates to generate Kubernetes manifests.
* Sends the manifests to the Kubernetes API server.
* Tracks the release's state (versions, configurations) in the cluster.

`helm uninstall` deletes all resources tied to the release in one command.
It is similar to `kubectl delete` but that requires manually deleting each resource or using a YAML file.

`helm list --all-namespaces` can show all deployed items.
