# Docker Practices

## Start Docker
Download from official website and install it.

Use **Docker Desktop Deb** version.

Example run as a check to success of installation:
```bash
docker run -d -p 80:80 docker/getting-started
```

Cmd docker start (tested on ubuntu):
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

### Base Image
Every container should start from a `scratch` image, aka base/parent image, such as

```dockerfile
# syntax=docker/dockerfile:1
FROM scratch
ADD hello /
CMD ["/hello"]
```

### Dockerfile and Syntax

```Dockerfile
# The FROM instruction initializes a new build stage and sets the Base Image for subsequent instructions. 
FROM ImageName

# Environment variable substitution will use the same value for each variable throughout the entire instruction. 
ENV abc=hello

# Similar to ENV, but ARG is mutable while ENV is immutable
ARG efg=world

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

# Add key-value pairs to this container
LABEL <key>=<value> <key>=<value> <key>=<value> ...
```

### Syntax Explained

* `FROM ImageName`

`FROM` creates a layer from the `ImageName` such as `ubuntu:22.04`.

* `RUN <command>` vs `CMD command`

`RUN` is an image build step, the state of the container after a RUN command will be committed to the container image. A Dockerfile can have many RUN steps that layer on top of one another to build the image.

`CMD` is the command the container executes by default when you launch the built image.
The `CMD` can be overridden when starting a container with `docker run $image $other_command`.

* `ADD <src> <dest>` vs `COPY <src> <dest>`

`COPY` simply copies files/directories from user's host machine to Docker image.

`ADD` besides copying files/directories, can download from URl and extract zip/tar to Docker image.

* `LABEL` vs `ENV` vs `ARG`

`LABEL` (for example, `LABEL version="1.0"`) sets image-level metadata, so it will be the same for all containers created from this image.

It can be used to provide human-readable texts such as below use cases.

```Dockerfile
LABEL version="1.0"
LABEL description="This is a hello world project."
```

`ENV` and `ARG` are used during image building for other instructions, such as variable substitution.

```Dockerfile
ENV binName=hello
CMD $binName
```

`ENV` is immutable.
For example, `ENV_VAR` cannot be updated.

```Dockerfile
ENV ENV_VAR=1
ENV ENV_VAR=2
```

`ARG` is same as `ENV` except that it is mutable.
For example, below `ARG_VAR` can be successfully updated.

```Dockerfile
ARG ARG_VAR=1
ARG ARG_VAR=2
```

* `EXPOSE`

It is used for port listening.

## Docker Cmd

* `docker run <ImageName>` - This command is used to start a new Docker container from an image. `-it` option indicates stdin and psuedo-TTY terminal. In other words, it provides an interative terminal.
* `docker ps` - This command is used to list all the running Docker containers. `-a` option displays history containers
* `docker stop <ImageName>` - This command is used to stop a running container.
* `docker rm <ImageName>` - This command is used to remove a Docker container.
* `docker images` - This command is used to list all the Docker images that are currently available on your system.
* `docker pull` - This command is used to download a Docker image from a registry.
* `docker exec <ContainerName>` - This command is used to execute a command in a running container.

Example:
Pull and launch a container from the image `ubuntu`.
```bash
# start docker (tested in ubuntu, in Mac or Windows, can just start docker from a Desktop version)
service docker start

# download the image
docker pull ubuntu

# get the image run in a container
docker run -it --name yuqi_ubuntu_terminal ubuntu 
```

Open in another terminal.
```bash
# check the recently started running container
docker ps | grep yuqi_ubuntu_terminal

# try running a cmd in the ubuntu container
docker exec yuqi_ubuntu_terminal bash -c "echo 'hello world'"

# should see the container stopped in a few secs
docker stop yuqi_ubuntu_terminal
```

### Volume

 `docker volume create [OPTIONS] [VOLUME]`

### Port Mapping

## Check Docker

Check Docker
```bash
### Check if there is a pid running docker
sudo service docker status

### Check docker which socket it listens to
ps -elf | grep docker

### Check version
docker version
```

If multiple dockers are installed, you can config docker to listen to another socket
```bash

```