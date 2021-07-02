FROM ubuntu:rolling
MAINTAINER Karolina Sienkiewicz <sienkiewicz2k@gmail.com

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update 

RUN apt-get install -y \
    python3 \
    python3-dev \
    python3-pip

RUN pip3 install --upgrade cython
RUN pip3 install python-sumo==0.2.7

CMD ["bash"]
