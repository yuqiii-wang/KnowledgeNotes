# K8s Networking

Every Pod in a cluster gets its own unique cluster-wide IP address.

## Service

For example, suppose you have a set of Pods where each listens on TCP port 9376 and contains a label `app=MyApp`.

Kubernetes assigns this Service an IP address

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: MyApp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 9376
```