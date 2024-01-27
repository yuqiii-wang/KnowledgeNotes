## build image
cd webapp-flask
docker build --tag webapp-flask-helloworld-docker .

## make sure image is built successfully and executable
docker image list | grep webapp-flask-helloworld-docker
docker run webapp-flask-helloworld-docker

## make sure minikube is started
minikube start

## load docker image to k8s (this may take a few minutes if image is large)
minikube image load webapp-flask-helloworld-docker:latest

## apply k8s manifest
cd ..
kubectl apply -f manifest-deployment.yaml
kubectl apply -f manifest-service.yaml

## show what url to access the app from load balancer
minikube service hello-world-service --url

## check services
minikube service list
# |----------------------|---------------------------|--------------|---------------------------|
# |      NAMESPACE       |           NAME            | TARGET PORT  |            URL            |
# |----------------------|---------------------------|--------------|---------------------------|
# | default              | hello-world-service       | http/80      | http://192.168.49.2:30213 |
# | default              | kubernetes                | No node port |                           |
# | kube-system          | kube-dns                  | No node port |                           |
# | kubernetes-dashboard | dashboard-metrics-scraper | No node port |                           |
# | kubernetes-dashboard | kubernetes-dashboard      | No node port |                           |
# |----------------------|---------------------------|--------------|---------------------------|

## 
kubectl get services
# NAME                  TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
# hello-world-service   LoadBalancer   10.107.136.97   <pending>     80:30213/TCP   5m13s
# kubernetes            ClusterIP      10.96.0.1       <none>        443/TCP        2m37s

## ingress
kubectl apply -f manifest-ingress.yaml
kubectl describe ingress minimal-hello-world-ingress
# kubectl describe ingress minimal-hello-world-ingress
# Name:             minimal-hello-world-ingress
# Labels:           <none>
# Namespace:        default
# Address:
# Ingress Class:    <none>
# Default backend:  <default>
# Rules:
#   Host                    Path  Backends
#   ----                    ----  --------
#   hellowworld.local.test
#                           /   hello-world-service:80 (10.244.0.31:8080,10.244.0.33:8080,10.244.0.34:8080)
# Annotations:              ingressclass.kubernetes.io/is-default-class: true
# Events:                   <none>