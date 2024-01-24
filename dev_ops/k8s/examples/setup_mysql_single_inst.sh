## Apply the manifest
kubectl apply -f manifest-mysql-single-deployment.yaml
kubectl apply -f manifest-mysql-single-volume.yaml

## wait for the mysql is set up and running, then create an interactive mysql client to talk to the mysql server.
## the `mysql-client` will be demised (as set by `--rm --restart=Never`) once exited
sleep 30
kubectl run -it --rm --image=mysql:5.6 --restart=Never mysql-client -- mysql -h mysql -ppassword
