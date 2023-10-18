FROM public.ecr.aws/w6p6i9i7/aws-efa-nccl-rdma:base-cudnn8-cuda11-ubuntu20.04

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true


RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y libgl1 git

RUN rm -rf /var/lib/apt/lists/*

RUN mkdir /code
# add conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P /code/
RUN chmod 777 /code/Miniconda3-latest-Linux-x86_64.sh
RUN /code/Miniconda3-latest-Linux-x86_64.sh -b -p /code/miniconda
ENV PATH="/code/miniconda/bin:${PATH}"

RUN groupadd miniconda
RUN chgrp -R miniconda /code/miniconda/ 
RUN chmod 770 -R /code/miniconda/ 

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
RUN adduser user miniconda

# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
	PYTHONUNBUFFERED=1 

RUN conda install python=3.9

RUN pip3 install setuptools-rust

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir Cython

RUN pip3 install imagecodecs

RUN pip3 install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115

COPY ./code/entrypoint.py /usr/local/bin/entrypoint.py

USER root
RUN mkdir -p /home/package && \
    mkdir -p /home/model-server/

RUN cd /home/package && \
    git clone https://github.com/NASA-IMPACT/hls-foundation-os.git && \
    git config --global --add safe.directory /home/package/hls-foundation-os && \
    cd /home/package/hls-foundation-os && \
    git checkout 9cdb612 && \
    pip3 install -e . --user

USER root
COPY ./code/requirements.txt /opt/code/requirements.txt

USER user
RUN pip3 install -r /opt/code/requirements.txt

RUN pip3 install -U openmim && \
    mim install mmengine==0.7.4 && \
    mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html

ENV CUDA_VISIBLE_DEVICES=0,1,2

ENV CUDA_HOME=/usr/local/cuda

ENV FORCE_CUDA="1"

USER root
COPY ./code/ /home/model-server/

# Define an entrypoint script for the docker image
ENTRYPOINT ["python", "/usr/local/bin/entrypoint.py", "serve"]

# Define command to be passed to the entrypoint
# CMD [""]