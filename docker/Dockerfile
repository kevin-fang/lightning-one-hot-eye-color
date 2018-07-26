FROM ubuntu:latest
MAINTAINER Kevin Fang <kfang@curii.com>
USER root

# install necessary dependencies
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y wget \ 
	gcc \
        libbz2-dev \
        liblzma-dev \
        dtrx \
	zlib1g-dev \
        make \
        python \
        python-pip \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran \
        g++ && \
    # install HTSLib for tile searching
    wget --output-document=htslib-1.5.tar.bz2 https://github.com/samtools/htslib/releases/download/1.5/htslib-1.5.tar.bz2 && \
    dtrx htslib-1.5.tar.bz2 && \
    cd htslib-1.5 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \ 
    rm ./htslib-1.5.tar.bz2 && \
    rm -rf htslib-1.5

# install python specific packages
RUN pip install --upgrade pip==9.0.3 && \ 
    pip install scikit-learn==0.19.1 \
        matplotlib \
        pandas \
        scipy \
        numpy \
        ipython[all]
