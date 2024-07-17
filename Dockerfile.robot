FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3

RUN apt-get update
RUN apt-get install -y vim git
RUN DEBIAN_FRONTEND=oninteractive apt-get install -y wget unzip
RUN python3 -m pip install --upgrade pip

# PCI
RUN pip3 install wheel --force-reinstall
RUN pip3 install setuptools --force-reinstall
RUN pip install dill pymaxflow

# kapao
RUN pip3 install matplotlib numpy Pillow PyYAML scipy tqdm gdown tensorboard seaborn pandas
RUN pip3 install Cython pycocotools thop pytube imageio


WORKDIR /workspace