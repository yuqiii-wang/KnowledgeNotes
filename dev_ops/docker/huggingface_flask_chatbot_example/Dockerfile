FROM ubuntu:20.04

WORKDIR /code

###### miniconda3 env
COPY external_libs/Miniconda3-py311_23.10.0-1-Linux-x86_64.sh /code/
RUN mkdir -p '/root/miniconda3/pkgs/'
RUN bash Miniconda3-py311_23.10.0-1-Linux-x86_64.sh -b -u
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN . ~/.bashrc
RUN echo "Running $(conda --version)" && conda init bash

###### Essentials and Cuda
COPY external_libs/cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb /code/
# RUN conda install make -y
# RUN conda install -c "nvidia/label/cuda-11.8.0" cuda -y
RUN dpkg -i cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb 


####### PyTorch, HuggingFace, and Flask
COPY ./requirements.txt /code/requirements.txt
# RUN mkdir -p /code/pytorch-main
# COPY external_libs/pytorch-main/* /code/
# RUN conda install cmake ninja -y
# Run this command from the PyTorch directory after cloning the source code using the “Get the PyTorch Source“ section below
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet
# RUN conda install mkl mkl-include
# CUDA only: Add LAPACK le/ for the GPU if needed
# RUN conda install -c pytorch magma-cuda118 
# RUN export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# RUN python /code/pytorch-main/setup.py develop

####### App
RUN mkdir -p /code/flask_server
RUN mkdir -p /code/google/flan-t5-small
COPY flask_server/* /code/flask_server
COPY external_libs/google/flan-t5-small/* /code/google/flan-t5-small
WORKDIR /code/flask_server


####### Set up a new user to run app (not root user out of security concerns)
# RUN useradd -m -u 1000 user
# RUN mkdir -p $HOME/flask_server
# USER user
# ENV HOME=/home/user \
# 	PATH=/home/user/.local/bin:$PATH
# WORKDIR $HOME/flask_server
# RUN cp -r /code/flask_server $HOME
# COPY --chown=user . $HOME/flask_server

CMD ['python', 'app']