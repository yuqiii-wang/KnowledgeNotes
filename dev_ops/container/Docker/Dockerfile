# FROM creates a layer from the ubuntu:22.04 Docker image.
FROM ubuntu:22.04

# COPY adds files from your Docker client's current directory.
COPY . /app

# RUN builds your application with make.
RUN make /app

# CMD specifies what command to run within the container.
CMD python /app/app.py