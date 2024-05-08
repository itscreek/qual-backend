FROM ubuntu:jammy

RUN apt update && apt upgrade -y && \
    apt install -y python3 python3-pip vim curl git

RUN mkdir /workspace

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt