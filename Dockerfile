FROM nvidia/cuda:10.1-cudnn8-runtime-ubuntu16.04

RUN apt -yy update
RUN apt install -yy --no-install-recommends python3 python3-pip
RUN pip install --no-cache-dir torch torchvision scipy

WORKDIR sim_matching/
COPY main.py .
COPY main_sim_final.py .
COPY config.py .
COPY main_robust.py .
ADD networks /sim_matching/networks
ADD checkpoint /sim_matching/checkpoint
 
ENTRYPOINT ["python3","main.py"]




