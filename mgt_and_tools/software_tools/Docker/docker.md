# Docker Cmds

## Start Docker
```bash
# start docker
service docker start
# check docker
sudo service docker status
```

To debug after failing to run `sudo service docker start`
```baah
sudo dockerd --debug
```

Docker might fail to start for no permission for `/etc/docker/daemon.json`.
```bash
> /etc/docker/daemon.json
sudo cat <<EOF>> /etc/docker/daemon.json
{}
EOF
```

### Dockerfile and Syntax

```Dockerfile
# The FROM instruction initializes a new build stage and sets the Base Image for subsequent instructions. 
FROM ImageName

# Environment variable substitution will use the same value for each variable throughout the entire instruction. 
ENV abc=hello

# (shell form, the command is run in a shell, which by default is /bin/sh -c on Linux or cmd /S /C on Windows)
RUN <command> 
# (exec form)
RUN ["executable", "param1", "param2"]

# There can only be one CMD instruction in a Dockerfile. 
# (exec form, this is the preferred form)
CMD ["executable","param1","param2"] 
# (as default parameters to ENTRYPOINT)
CMD ["param1","param2"] 
# (shell form)
CMD command param1 param2 

# The EXPOSE instruction informs Docker that the container listens on the specified network ports at runtime.
EXPOSE <port> [<port>/<protocol>...]

# The ADD instruction copies new files, directories or remote file URLs from <src>s and adds them to the filesystem of the image at the path <dest>.
ADD [--chown=<user>:<group>] <src>... <dest>
ADD [--chown=<user>:<group>] ["<src>",... "<dest>"]

# The COPY instruction copies new files or directories from <src>s and adds them to the filesystem of the container at the path <dest>.
COPY [--chown=<user>:<group>] <src>... <dest>
COPY [--chown=<user>:<group>] ["<src>",... "<dest>"]

```


### Docker vs Docker-Compose