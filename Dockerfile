FROM nvidia/cuda:10.1-cudnn8-runtime-ubuntu18.04

RUN apt -yy update
RUN apt install -yy --no-install-recommends python3 python3-pip
RUN pip install torch torchvision scipy
