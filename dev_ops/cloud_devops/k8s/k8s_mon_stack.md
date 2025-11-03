# K8S Monitoring Stack

Introduce K8S Grafana Stack (Prometheus + Loki).

Grafana is a dashboard with provisioned data from prometheus (typically for general http requests) or from Loki (typically for log text search).

## Prometheus in K8S

### General Prometheus Intro

Prometheus provides below monitoring functions:

* **Collect** metrics from configured targets at regular intervals.
* **Store** these metrics efficiently with a timestamp and a set of key-value labels.
* **Enable** to query this data using a flexible query language (PromQL).
* **Trigger** alerts if certain conditions, defined in rules, are met.

Every target that Prometheus monitors is expected to expose its metrics on an HTTP endpoint, typically called `/metrics`, in a specific text format. Prometheus periodically "scrapes" (fetches) the data from this endpoint.

### Prometheus Custom Resource Definitions (CRDs)

In Kubernetes, CRDs extend the API to recognize custom, domain-specific resources.
The `PrometheusRule` is one such custom resource (most common K8S-builtin resources are `Pod`, `Deployment`, `StatefulSet`, `DaemonSet`, `Job`, `ReplicaSet`, `Service`, `Ingress`, `PersistentVolume`, `PersistentVolumeClaim`, `ConfigMap`, `Namespace`).

### Special Labels for Prometheus's Service Discovery

Such label primary purpose is to act as temporary, internal "control signals" or variables during the service discovery and relabeling phases.
Most of them have a prefix of `__...`, and are discarded before the metrics are stored in the database.

#### The Scrape Target Labels

These labels directly control how the final HTTP request to a target is constructed.

|Label Name|Purpose|Default Value|
|:---|:---|:---|
|`__address__`|The host:port address **of the target to scrape**. This is the most critical label.|Comes from the service discovery mechanism (e.g., a pod's IP and port).|
|`__scheme__`|The protocol to use for the scrape.|http|
|`__metrics_path__`|The URL path to scrape for metrics on the target.|`/metrics`|
---

#### Service Discovery "Meta" Labels (`__meta_...`)

These are read-only labels created by Prometheus's service discovery mechanisms. They expose all the metadata Prometheus has learned about a potential target from its source (like the Kubernetes API, AWS EC2 API, Consul, etc.).

* `__meta_kubernetes_namespace`: The namespace of the discovered object
* `__meta_kubernetes_pod_name`: The name of the pod
* `__meta_kubernetes_pod_ip`: The pod's IP address
* `__meta_kubernetes_pod_container_name`: The name of a container within the pod.
* `__meta_kubernetes_pod_ready`: The readiness status of the pod

Also provided custom labels and annotations

* `__meta_kubernetes_pod_label_<labelname>`: Exposes a pod's Kubernetes label. The `<labelname>` is converted to be Prometheus-compatible (e.g., a Kubernetes label of `app.kubernetes.io/name: "my-app"` becomes `__meta_kubernetes_pod_label_app_kubernetes_io_name: "my-app"`).
* `__meta_kubernetes_pod_annotation_<annotationname>`: Exposes a pod's Kubernetes annotation. This is commonly used to enable scraping on a per-pod basis (e.g., `__meta_kubernetes_pod_annotation_prometheus_io_scrape: "true"`).

#### Parameter Labels for Probing

* `__param_<name>`|Converts this label into a URL query parameter `&<name>=<value>` in the final scrape request.

Work flow:

1. A `relabel_config` rule takes the real target address and puts it into the `__param_target` label e.g., `__param_target: "https://my-app.com"`
2. Another rule sets the `__address__` to point to the `blackbox-exporter` app itself, e.g., `__address__: "blackbox-exporter:9115"`
3. The job config sets metrics_path: `/probe`.

Prometheus then constructs the URL: `http://blackbox-exporter:9115/probe?target=https%3A%2F%2Fmy-app.com`

### `scrape_configs`: From Discovery to HTTP Request

Example: a `my-app` Scrape (The `my-app` job)

Assumed an endpoint for a PostgreSQL pod is discovered at `10.96.7.8:5432`.

1. Discovery: The `kubernetes_sd_configs` with role: endpoints discovers this target. Prometheus creates the initial meta labels:
    * `__address__: "10.96.7.8:5432"`
    * `__meta_kubernetes_service_name: "my-app"`
2. Relabeling:
    * Copy the value of `__address__` (10.96.7.8:5432) into a new label called `__param_target`.
    * Completely replaces the `__address__` label with the static value `my-app.<namespace>.svc.cluster.local:9115`.
3. Final HTTP Request Construction: Prometheus now assembles the request.
    * `__scheme__: http`
    * `__address__: blackbox-exporter.<namespace>.svc.cluster.local:9115`
    * `__metrics_path__`: `/probe` (defined by metrics_path: `/probe` in the job config)
And now for the special labels: any label prefixed with `__param_` is automatically converted into a URL parameter.
    * `__param_target`: `10.96.7.8:5432` -> becomes `&target=10.96.7.8:5432`
The job also has a static params: {module: [tcp_connect]} which becomes `&module=tcp_connect`.

Prometheus combines all of this to generate the final HTTP request:

```txt
GET http://my-app.<namespace>.svc.cluster.local:9115/probe?module=tcp_connect&target=10.96.7.8:5432
```

### `relabel_configs`: The "If-Then" Engine for Targets

IF the `source_labels` concatenated together match the `regex`, THEN perform the `action`.

1. `source_labels`: [The "IF" part] This is a list of existing label names (usually `__meta_` labels). Prometheus takes their values and joins them with a semicolon (`;`) to create a single string to test against.
2. `regex`: [The "MATCH" part] This is a regular expression that is applied to the string from `source_labels`. If it doesn't match, the action is typically skipped (unless the action is drop).
3. `action`: [The "THEN" part] This is what to do if the regex matches. The most common actions are:
    * `keep`: Keep the target only if the regex matches. Drop it otherwise.
    * `drop`: Drop the target if the regex matches. Keep it otherwise.
    * `replace`: It modifies or creates a target_label.
    * `labelmap`: It applies the regex to all label names and copies any matching label to a new label. For example, `regex: __meta_kubernetes_pod_label_(.+)` with replacement: `$1` will copy `__meta_kubernetes_pod_label_app: "my-app"` to a clean final label `app: "my-app"`.
4. `target_label`: [The "DESTINATION" for replace] The name of the label you want to create or overwrite.
5. `replacement`: [The "NEW VALUE" for replace] The new value for the target_label. It 2can be used to capture groups from regex here (e.g., `$1`, `$2`) to construct the new value dynamically.

### Full Example Explained

The below Prometheus config sets up scheduled GET requests to blackbox that delegates an health check to `my-app`.

```txt
GET http://blackbox-exporter.monitoring.svc:9115/probe?module=http_2xx&target=http%3A%2F%2Fmy-app%3A8080%2Fhealthz
```

1. Initial Target: Prometheus starts with the target defined in static_configs: `http://my-app:8080/healthz`. The special label `__address__` is set to this value.
2. Relabeling:
    * The first rule copies the value of `__address__` into the `__param_target` label -> The `__param_` prefix is a command to Prometheus turning this label into a URL query parameter, e.g., `&target=http%3A%2F%2Fmy-app%2Fhealthz`
    * The second rule copies that same value into the permanent `instance` label, so metrics (on grafana) will clearly show what was probed. The the final metric will be stored with the label `{instance="http://my-app/healthz", ...}`.
    * The third rule completely redirects the scrape by overwriting the `__address__` label to `blackbox-exporter.monitoring.svc:9115`.
3. Final HTTP Request: Prometheus constructs and sends the scheduled GET request to the Blackbox Exporter

where the `instance` label is a permanent, default label that Prometheus adds to every metric it scrapes.
Its purpose is to identify the specific target from which the metrics were collected.

```json
{
  "global": {
    "scrape_interval": "15s",
    "evaluation_interval": "15s",
    "external_labels": {
      "cluster": "production-cluster"
    }
  },
  "alerting": {
    "alertmanagers": [
      {
        "static_configs": [
          {
            "targets": [ "alertmanager.monitoring.svc:9093" ]
          }
        ]
      }
    ]
  },
  "rule_files": [
    "/etc/prometheus/rules/*.rules.yml"
  ],
  "scrape_configs": [
    {
      "job_name": "my-app-health-probe",
      "metrics_path": "/probe",
      "params": {
        "module": [ "http_2xx" ]
      },
      "static_configs": [
        {
          "targets": [ "http://my-app:8080/healthz" ]
        }
      ],
      "relabel_configs": [
        {
          "source_labels": [ "__address__" ],
          "action": "replace",
          "target_label": "__param_target",
          "replacement": "$1"
        },
        {
          "source_labels": [ "__param_target" ],
          "action": "replace",
          "target_label": "instance",
          "replacement": "$1"
        },
        {
          "action": "replace",
          "target_label": "__address__",
          "replacement": "blackbox-exporter.monitoring.svc:9115"
        }
      ]
    }
  ]
}
```

### PromQL Intro

PromQL (Prometheus Query Language) is a query language designed specifically for time-series data that are performed calculations on the metrics that Prometheus collects.

To filter by condition, add in curly braces `{}`.

* `=` : Exactly equal.
* `!=` : Not equal.
* `=~` : Regex match.
* `!~` : Does not regex match.

To select a range of data points, add in square brackets `[]`.
Common Durations:

* `s` (seconds)
* `m` (minutes)
* `h` (hours)
* `d` (days)

To apply aggregation, there are

* `rate(...)` calculates the per-second average rate of increase of a time series over a given time window.
* `increase(...)` calculates the total increase in a counter over a given time window.

Example: Calculate the average per-second rate of HTTP requests with a 200 status code over the last 5 minutes.

```txt
rate(http_requests_total{code="200"}[5m])
```

To apply groupBy, use `by`.

For example, http load results are grouped by `node` so that audit can clearly see which nodes have high traffic.

```txt
sum by (node) rate(http_requests_total{code="200"}[5m])
```

## Loki, TSDB, and Data Persistence

Loki is used for logging and checking.
Generally speaking, there are

* TSDB (Time-Series DB) for INDEX store
* The actual compressed log data is stored as CHUNKs
* Promtail as log collection agent
* Grafana to Promtail

### Low-Level Implementation of TSDB Index Mapping to Chunks

TSDB (not to get confused with TimeScaleDB) is heavily optimized for the specific workload of time-series data: a very high rate of small writes, with reads that scan across a time range for a specific set of labels.

#### When new log data arrives

When new log data arrives, its index is not immediately written to a permanent file on disk. Instead, it goes through two initial steps for speed and durability:

* Write-Ahead Log (WAL): Before anything else, the incoming data is written to a Write-Ahead Log on disk. The WAL is a sequential log of all operations. Its sole purpose is durability; if the server crashes, it can replay the WAL to reconstruct the in-memory data that hadn't yet been persisted.
* In-Memory Head Block: After being secured in the WAL, the data is written to an in-memory database called the Head block. This is where all recent data lives, making queries for the last few hours extremely fast as they are served directly from RAM.

#### Chunks and Memory Mapping

Within the Head block, the actual samples (timestamp/value pairs) for a given series are compressed into chunks. As these chunks age, TSDB employs a clever optimization:

* The "full" chunks are flushed to disk into a chunks_head directory and then memory-mapped (m-mapped).
* This means the data now resides on disk, but the operating system handles loading it into memory as if it were still there. This significantly reduces the RAM footprint of Loki/Prometheus while keeping recent data easily accessible.

#### Persistent Blocks

By default, every two hours, a process called "Head Compaction" runs.

It takes the oldest data from the Head block (both in-memory and m-mapped chunks) and writes it to the disk as a new, immutable persistent block.
Once the block is successfully written, the corresponding data can be purged from the Head block and the WAL can be truncated.

#### On-Disk Block Structure

Each persistent block is a self-contained database for its time range (e.g., two hours). It is a directory on the filesystem containing:

* `chunks`: One or more files that contain all the compressed log/metric data for that time period.
* `index`: A highly optimized index file. This file contains several key parts:
    * Postings (Inverted Index): The core of the index. It maps a label value (e.g., app="api") to a list of all the series IDs that have that label. This is what makes lookups so fast.
    * Series List: A list of all unique series in the block, their labels, and pointers to their data in the chunks files.
    * Symbol Table: A dictionary that maps recurring strings (label names and values) to numeric IDs to save space.
* `meta.json`: A small file containing metadata about the block, such as the minimum and maximum timestamps of the data it contains.
* `tombstones`: Records deletions. Data is not deleted immediately but is marked in a tombstone file and removed during a later compaction.

#### Background Compaction

Having accumulated a large number of two-hour blocks, a background Compactor process runs.

* It merges multiple smaller blocks (e.g., several 2-hour blocks) into a single, larger block.
* Compaction also handles data retention, removing blocks that are older than the configured retention period.

### Typical Loki Config in K8S

#### Listening Port

This tells the Loki process to listen for all incoming HTTP traffic (API requests for both writing logs and querying logs) on port 3100.

This value directly corresponds to the `containerPort: 3100` specified in Loki Deployment YAML.

If there are multiple loki instances, set up `grpc_listen_port: 9095` for internal communication between Loki components.

```yaml
server:
  http_listen_port: 3100
```

#### Ingester

The `ingester` is the component responsible for the "write path". It receives streams of log data, groups them into "chunks" in memory, and then flushes these chunks to the long-term storage backend.

```yaml
ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
  # ...
```

where

* `ring`: In a distributed setup, the ring is a key-value store that helps coordinate which ingester is responsible for which log streams.
* `kvstore.store: inmemory`: It tells Loki to manage the ring in its own memory. It only works for ONE replica.
* `replication_factor: 1`: This means each log stream is only handled by one ingester.

```yaml
distributor:
  ring:
    kvstore:
      store: memberlist
```

#### Store Schema and Config

* `store: tsdb`: It tells Loki to use the Time Series Database (TSDB) as its indexing mechanism.
* `object_store: filesystem`: `filesystem` means it will use the local filesystem of the container to store the actual compressed log data (the "chunks").
* `tsdb_shipper`: This block is active given the chosen store: tsdb. It controls the TSDB index.

```yaml
schema_config:
  configs:
    - from: 2020-10-24
      store: tsdb
      object_store: filesystem
      schema: v12
      index:
        prefix: index_
        period: 24h
storage_config:
  tsdb_shipper:
    active_index_directory: /data/tsdb-index
    cache_location: /data/tsdb-cache
  filesystem:
    directory: /data/chunks
```

#### Logging Resource Control

`reject_old_samples: true` and `reject_old_samples_max_age: 168h`: This tells Loki to reject any log lines that have a timestamp older than 7 days (168 hours). This prevents a misconfigured client from sending very old logs and forcing Loki to create index files for a long-past date.

```yaml
limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
```

#### Multi-Instance Loki Config

For multi-instance Loki env, need to set up additional ports and config for internal communication.

```yaml
server:
  http_listen_port: 3100
  grpc_listen_port: 9095 # for internal communication between Loki components.
distributor:
  ring:
    kvstore:
      store: memberlist
ingester:
  lifecycler:
    address: ${POD_IP} # # Each Loki instance needs a unique address to register itself in the ring.
    ring:
      kvstore:
        store: memberlist # CHANGED from 'inmemory'
      replication_factor: 3 # With multiple replicas, we can replicate data for safety.
memberlist:
  # This is the name of the Kubernetes Headless Service we will create.
  # It allows pods to discover each other via DNS.
  join_members:
    - loki-headless.my-loki-env.svc.cluster.local
```

### Promtail: The Log Collection Agent

Promtail is the agent responsible for collecting logs and sending them to Loki.

Promtail config includes

* `server`: This section configures Promtail's own web server, which can be used for health checks and exposing its own metrics.
* `positionS`: this tells Promtail where to save the file that tracks its progress in reading logs.
*  `clients`: defines the Loki instance(s) to which Promtail will send logs.

About the scrape config:

* `kubernetes_sd_configs`: This tells Promtail to use its built-in Kubernetes Service Discovery.
* `role: pod`: This specifically instructs Promtail to connect to the Kubernetes API and ask for a list of all running pods. For each pod it finds, it creates a "target" and automatically attaches a large amount of metadata about that pod (its name, namespace, labels, node, etc.).

Recall that the `replace` action is to copy `source_labels` (the `__meta_*`) to a special internal label called `__host__`, `__path__`, etc.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: promtail-config
  namespace: my-loki-env
data:
  promtail-config.yaml: |
    server:
      http_listen_port: 9080
      grpc_listen_port: 0

    positions:
      filename: /tmp/positions.yaml

    clients:
      - url: http://loki:3100/loki/api/v1/push

    scrape_configs:
    - job_name: kubernetes-pods
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_node_name]
        target_label: __host__
      - action: replace
        source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
      - action: replace
        source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - action: replace
        source_labels: [__meta_kubernetes_pod_container_name]
        target_label: container
      - replacement: /var/log/pods/*$1/*.log
        source_labels: [__meta_kubernetes_pod_uid, __meta_kubernetes_pod_container_name]
        target_label: __path__
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: promtail
  namespace: my-loki-env
  labels:
    app: promtail
spec:
  selector:
    matchLabels:
      app: promtail
  template:
    metadata:
      labels:
        app: promtail
    spec:
      serviceAccountName: promtail
      containers:
      - name: promtail
        image: grafana/promtail:latest
        args:
        - "-config.file=/etc/promtail/promtail-config.yaml"
        volumeMounts:
        - name: promtail-config
          mountPath: /etc/promtail
        - name: pods-logs
          mountPath: /var/log/pods
          readOnly: true
      volumes:
      - name: promtail-config
        configMap:
          name: promtail-config
      - name: pods-logs
        hostPath:
          path: /var/log/pods
```

where, to understand how promtail collects logs, there are

* `volumeMount` path: `/etc/promtail`: This volume's job is purely to deliver the configuration file (`promtail-config.yaml` from `ConfigMap`) into the Promtail container's filesystem.
* `volumeMount` path: `/var/log/pods`: Allows Promtail to access log files from **all other pods** on the **same node**.

#### Promtail Deployment in Cluster

##### Promtail and Filesystem: No Need of New Persistent Volume (PV)

When application pod runs, it writes logs to standard output (`stdout`).

1. The container runtime on the Node captures this output.
2. The Kubelet (the main Kubernetes agent on the Node) takes this captured output and writes it to a file on the Node's own filesystem.
3. The standard location for these files is `/var/log/pods`. The directory structure is very specific:

```txt
# On the Node's filesystem
/var/log/pods/
└── <namespace>_<pod_name>_<pod_uid>/
    └── <container_name>/
        ├── 0.log
        └── 1.log (if rotated)
```

As a result, there is no need of launching a user-managed PV, but let all `stdout` to the default paths.

##### From Promtail to Loki and Grafana

|Component|Role|Kubernetes Object|Number of Pods in a Multi-Node Cluster|Why?
|:---|:---|:---|:---|:---|
|Promtail|Log Collector|`DaemonSet`|Multiple (**One per node**)|Must run on every node to access that node's local log files (`/var/log/pods`).|
|Loki|Log Storage & Indexing|`Deployment` or `StatefulSet`|Centralized (One or more replicas, but not one per node)|Acts as a central destination for all Promtail agents to send logs to.|
|Grafana|Visualization & Querying|`Deployment`|Centralized (Usually **one pod**)|Acts as a single web UI for users to query Loki and view dashboards.|

All Promtail pods (from all nodes) are configured to point to the single Kubernetes `Service` that exposes Loki `Deployment` or `StatefulSet`.

```yaml
clients:
  - url: http://loki:3100/loki/api/v1/push
```
 
## Alert Manager for Grafana

The data flow is Grafana Alert -> Prometheus -> Alertmanager -> Notification.

For example, to trigger CPU high usage alert fired when > 80%, 
let

* `"refId": "A"` be `"expr": "sum by (mode) (rate(node_cpu_seconds_total{...}[1m]))"`
* `"refId": "B"` be `"expr": "vector(80)"`

then define `"refId": "C"` as the boolean condition.

```json
{
  "datasource": "-- Expressions --",
  "type": "math",
  "expression": "$A > $B",
  "refId": "C"
}
```

Then define the alert in Grafana evaluating the `C`.

```json
{
  "__inputs": [],
  "__requires": [],
  "annotations": {
    "list": []
  },
  ...,
  "alert": {
    "name": "High CPU Usage Alert",
    "message": "The CPU usage is above 80%.",
    "for": "5m",
    "conditions": [
      {
        "type": "query",
        "evaluator": { "params": [0], "type": "gt" }, // is above 0
        "query": { "params": ["C"] } // Evaluate query C
      }
    ],
    "notifications": [], // Leave empty for Grafana 8+ managed alerts
    "annotations": {
      "summary": "High CPU on instance {{ $labels.instance }}",
      "description": "CPU usage for job {{ $labels.job }} is at {{ $values.A }}%, which is above the 80% threshold."
    },
    "labels": {
      "severity": "critical"
    }
  }
}
```

where in `"alert": { ... }`

* `conditions`: This tells the alert to trigger when the result of query C is greater than 0.
* `for: "5m"`: The condition must be true for 5 continuous minutes before the alert becomes "Firing". This prevents flapping.

### Alert Manager Notification

The above `"alert": { ... }` is just to set up alert in grafana dashboard; it has no notification.
Below set up notification via an assumed local email server `smtp_smarthost: 'localhost:25'` specified in config map `data.alertmanager.yml`.

The image `prom/alertmanager:latest` takes `alertmanager.yml` as config to run.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: monitoring
data:
  alertmanager.yml: |-
    global:
      # The default SMTP server, if you were using email.
      # smtp_smarthost: 'localhost:25'
      # smtp_from: 'alertmanager@example.org'

    # The root route. All alerts enter here.
    route:
      # The default receiver for alerts that don't match any specific sub-routes.
      receiver: 'webhook-receiver'

      # Rules for grouping alerts. Alerts with the same labels here will be in a single notification.
      group_by: ['alertname', 'job']
      
      # How long to wait before sending an initial notification.
      group_wait: 30s
      
      # How long to wait before sending a notification about new alerts for an already active group.
      group_interval: 5m
      
      # How long to wait before re-sending a notification if an alert is still firing.
      repeat_interval: 4h

    # A list of notification receivers.
    receivers:
    - name: 'webhook-receiver'
      webhook_configs:
      # A webhook is a generic way to send alerts. Use a site like webhook.site to get a test URL.
      # In a real scenario, this would be the URL for Slack, Microsoft Teams, etc.
      - url: 'https://webhook.site/your-unique-test-url'
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alertmanager
  template:
    metadata:
      labels:
        app: alertmanager
    spec:
      containers:
      - name: alertmanager
        image: prom/alertmanager:latest
        args:
          - "--config.file=/etc/alertmanager/alertmanager.yml"
        ports:
        - containerPort: 9093
        volumeMounts:
        - name: config-volume
          mountPath: /etc/alertmanager/
      volumes:
      - name: config-volume
        configMap:
          name: alertmanager-config
---
apiVersion: v1
kind: Service
metadata:
  name: alertmanager
  namespace: monitoring
spec:
  selector:
    app: alertmanager
  ports:
  - port: 9093
    targetPort: 9093
---
```

where how alert is sent is in `route.receiver`.

1. Alertmanager receives the alert.
2. It processes the alert through its `route`.
3. The route directs the alert to `receiver`.
4. Depending on what receiver:
  * The `do-nothing` receiver has no configured actions, so it simply acknowledges the alert and then discards it.
  * The `'webhook-receiver'` sends an HTTP POST request to the URL specified in url: `'https://webhook.site/your-unique-test-url'`.

## Grafana

Grafana is the config component to define monitoring dashboard style and datasources by which display rules are set up.

```json
{
  "__inputs": [],
  "__requires": [],
  "annotations": {
    "list": []
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": null,
  "links": [],
  "panels": [
    ...
  ],
  "refresh": "10s",
  "schemaVersion": 30,
  "style": "dark",
  "tags": ["minimal", "cpu"],
  "templating": {
    ...
  },
  "time": {
    "from": "now-1h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Minimal CPU Dashboard",
  "uid": "minimal-cpu-dashboard-example",
  "version": 1,
  "alert": {
    ...
  }
}
```

where

* `"__inputs": []` and `"__requires": []`: These are internal metadata fields used by Grafana, primarily used to import dashboard that requires specific data source or plugin.

### Panel and Query (`panels`)

```json
"panels": [
  {
    "title": "CPU Usage",
    "type": "graph",
    "datasource": "Prometheus",
    "gridPos": { "h": 10, "w": 24, "x": 0, "y": 5 },
    "targets": [
      {
        "expr": "sum by (mode) (rate(node_cpu_seconds_total{instance=~\"job\"}[1m]))",
        "legendFormat": "{{mode}}",
        "refId": "A"
      },
      {
        "expr": "vector(80)",
        "refId": "B",
        "legendFormat": "Threshold"
      }
    ],
    "yaxes": [ { "format": "percent", "max": "100" } ],
    "stack": true,
    "percentage": true
  }
]
```

where

* `"type": "graph"`: This tells Grafana to render the data as a time-series graph (a line chart). Other common types include `"stat"` (for a single number), `"table"`, `"gauge"`, or `"barchart"`.
* `"datasource": "Prometheus"`: This name must match the one you set up in the grafana-datasources ConfigMap. Common sources can be `"Prometheus"` and `"loki"`
* `"gridPos": {...}`: The panel's size and position
* `"targets": [ ... ]`: A list of query results
  * `"expr": "sum by (mode) (rate(node_cpu_seconds_total{instance=~\"$node\",job=~\"$job\"}[1m]))"`: through `by` to group by `mode` as category that values are summed from calculation of the per-second average rate of increase of the CPU counter.
  * `{instance=~\"$node\",job=~\"$job\"}`: Interactive as dropdown to select on dashboard as filter
  * `"legendFormat": "{{mode}}"`: A template for formatting the names in the graph's legend.
  * `"refId": "A"` and `"refId": "B"`: Query result references
* `"stack": true` and `"percentage": true`: Styling options for the graph.

About `"refId": "A"` and `"refId": "B"`, for example to trigger an alert if CPU usage is high exceeded the threshold, it can writes

```json
{
  "datasource": "-- Expressions --",
  "type": "math",
  "expression": "$A > $B",
  "refId": "C"
}
```

### Interactivity and Templating (`templating`)

#### Prerequisite Knowledge - The *job*

##### The Jobs in Prometheus

In the world of Prometheus, a job is a label that is automatically attached to every metric collected from a group of targets. Its purpose is to identify the role or purpose of those targets.

```yaml
scrape_configs:
  # Job 1: Monitor all your Linux/macOS servers using Node Exporter
  - job_name: 'node_exporter'
    static_configs:
      - targets: ['server1:9100', 'server2:9100', 'server3:9100']

  # Job 2: Monitor all instances of your web application
  - job_name: 'webapp'
    static_configs:
      - targets: ['app1:8080', 'app2:8080']

  # Job 3: Monitor Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

##### Prometheus Job Consumption in Grafana

Grafana is a visualization tool that consumes the labels provided by Prometheus.
`job` is a special variable that hooks up Prometheus to Grafana.

For example, having defined three jobs in the above config `scrape_configs`, 

```txt
rate(node_cpu_seconds_total{job="$job", mode="system"}[5m])
```

the Grafana expression on `job="$job"` will have a dropdown menu at the top labeled `"Job"`. A user can:

* Select `"node_exporter"` to see the CPU usage for all servers.
* Select `"webapp"` to see metrics for all web applications (if they exposed a CPU metric).
* Select `"prometheus"` to see metrics for Prometheus itself.

The `job` can be further used for aggregation, where the grouping index is `job`.

For example, below shows the total number of HTTP requests per second across ALL instances, but separated by job.

```txt
sum by (job) (rate(http_requests_total[1m]))
```

#### Grafana Templating Example

`templating` in Grafana allows user to create interactive and dynamic dashboards.

Again as an example, having defined three jobs in the above config `scrape_configs`, Prometheus on user Grafana dropdown click responds with a list of all unique job names it has.
The dropdown options contain all unique jobs plus an `"ALL"` option.

```json
"templating": {
  "allValue": ".*",
  "current": {},
  "datasource": {
    "type": "prometheus",
    "uid": "YOUR_PROMETHEUS_DATASOURCE_UID"
  },
  "definition": "label_values(http_requests_total, job)",
  "description": "Filter by a specific job.",
  "hide": 0,
  "includeAll": true,
  "label": "Job",
  "multi": true,
  "name": "job",
  "options": [],
  "query": {
    "query": "label_values(http_requests_total, job)",
    "refId": "StandardVariableQuery"
  },
  "refresh": 1,
  "regex": "",
  "skipUrlSync": false,
  "sort": 1,
  "type": "query"
}
```

where

* `"allValue": ".*"`: When the `"All"` option is selected, this is the value that the $job variable will take. In this case, it's the regex .*, which means "match everything."
* `"name": "job"`: This is the internal name of the variable in panel queries, typically with a dollar sign prefix, like `$job`.
* `"label": "Job"`: This is the user-friendly text that will appear next to the dropdown menu on the dashboard.
* `"query": "label_values(node_exporter_build_info, job)"`: This is the core of the variable's definition. It's a Prometheus Query Language (PromQL) query that Grafana will execute to populate the dropdown list.
  * `label_values(...)` is a Prometheus function that returns a list of all unique values for a specific label.
  * `node_exporter_build_info` is the metric being queried.
  * `job` is the label whose values user wants to extract.

#### User Interaction on The Grafana Dropdown List

The Prometheus backend performs this PromQL (the `=~` is a regex match expression NOT equality evaluation):

```txt
sum by (job) (rate(http_requests_total{job=~"$job"}[1m]))
```

When user on the Grafana - Prometheus hookups are

* Single Selection: If a user selects `"webapp"` from the dropdown, Grafana replaces `$job` with `webapp`, and the query becomes `...{job=~"webapp"}[1m]`. The graph will show the request rate for just the `webapp` job.
* Multi-Selection: If a user selects `"webapp"` and `"api-gateway"`, Grafana automatically formats `$job` as a regex OR (`webapp|api-gateway`). The query becomes `...{job=~"webapp|api-gateway"}[1m]`, and the graph will show two lines, one for each job.
* "All" Selection: If a user selects "All", Grafana uses the `"allValue"` (`.*`). The query becomes `...{job=~".*"}[1m]`, which matches *all* jobs, and the graph will show the request rates for every available job, perfectly fulfilling the intent of your original query.
