FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ARG NB_USER="sagemaker-user"
ARG NB_UID="1000"
ARG NB_GID="100"

RUN apt update

RUN apt-get update && \
    apt-get install -y python3 python3-pip

RUN useradd --create-home --shell /bin/bash --gid "${NB_GID}" --uid ${NB_UID} ${NB_USER}

RUN apt clean

RUN rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install jupyterlab

RUN python3 -m pip install --upgrade urllib3==1.26.6

USER ${NB_UID}
CMD jupyter lab --ip 0.0.0.0 --port 8888 \
  --ServerApp.base_url="/jupyterlab/default" \
  --ServerApp.token='' \
  --ServerApp.allow_origin='*'
