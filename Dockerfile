# Author: Gaëtan Hadjeres

# ==== Build this image with the following command
# CIA superconditioning: aka. piano_inpainting_app:v3
# docker build -t piano_inpainting_app:v3 --build-arg GITHUB_TOKEN="$(cat /home/gaetan/.secrets/github_token)" --build-arg SSH_PRIVATE_KEY="$(cat /home/gaetan/.ssh/id_rsa)" --build-arg AWS_BUCKET_NAME="piano_event_performer_2021-09-03_18:40:31_finetune" .

# PIAv3
# docker build -t piano_inpainting_app:v3 --build-arg GITHUB_TOKEN="$(cat /home/gaetan/.secrets/github_token)" --build-arg SSH_PRIVATE_KEY="$(cat /home/gaetan/.ssh/id_rsa)" --build-arg AWS_BUCKET_NAME="piano_event_performer_2021-10-01_16:03:06_TOWER_32j" .  


# ===== Run with docker with
# docker run -e NVIDIA_VISIBLE_DEVICES=7 --rm -it -p 5000:8080 --gpus=0 piano_inpainting_app:v2 serve
# or 
# docker run -e NVIDIA_VISIBLE_DEVICES=7 --rm -it -p 5000:8080 --gpus=0 piano_inpainting_app:v3 serve

# Need to use nvidia-container-runtime to build docker with GPUs
# https://github.com/nvidia/nvidia-container-runtime#docker-engine-setup

FROM nvidia/cuda:11.1-devel-ubuntu20.04 as intermediate

LABEL maintainer hadjeres.g@gmail.com
    
RUN apt-get update -y && apt-get install -y git && apt-get install -y curl

ARG SSH_PRIVATE_KEY
ARG GITHUB_TOKEN
ARG RELEASE_NAME

RUN mkdir /root/.secrets/


RUN mkdir /root/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa && chmod 600 /root/.ssh/id_rsa
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# clone repos in the intermediate image
# We must use "clean" branch from DatasetManager 
# Merci Léo
RUN git clone --branch clean git@github.com:SonyCSLParis/DatasetManager.git /workspace/DatasetManager \
    && git clone git@github.com:Ghadjeres/fast-transformers.git /workspace/fast-transformers \
    && git clone git@github.com:SonyCSLParis/jazz-music21.git /workspace/jazz-music21

# PIAv3
RUN ls -l
RUN curl -vJLO -H "Authorization: token ${GITHUB_TOKEN}" "https://github.com/SonyCSLParis/CIA/archive/refs/tags/v${RELEASE_NAME}.tar.gz"
RUN tar -xvzf "CIA-${RELEASE_NAME}.tar.gz" && mv "CIA-${RELEASE_NAME}/" /workspace/CIA

FROM nvidia/cuda:11.1-devel-ubuntu20.04

COPY --from=intermediate /workspace /workspace

# install Anaconda
ENV PATH="/root/miniconda3/bin:$PATH"
RUN apt-get update -y \
    && apt-get install -y wget \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    &&  mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# create conda env
COPY environment.yml /workspace/environment.yml
RUN conda env create -f /workspace/environment.yml

# make all RUN commands use cia2 conda env
SHELL ["conda", "run", "-n", "cia2", "/bin/bash", "-c"]

WORKDIR /workspace

RUN apt-get install g++ -y
RUN conda install -c pytorch pytorch=1.6

RUN which python
# install all local python packages
WORKDIR /workspace/jazz-music21
RUN pip install -e .
WORKDIR /workspace/DatasetManager
RUN pip install -e .

WORKDIR /workspace/CIA

# RUN pip install -e .


# Download model
ARG AWS_BUCKET_NAME

RUN mkdir models && mkdir "models/${AWS_BUCKET_NAME}"

RUN wget "http://ghadjeres.s3.amazonaws.com/${AWS_BUCKET_NAME}/config.py" -P "models/${AWS_BUCKET_NAME}"
RUN wget "http://ghadjeres.s3.amazonaws.com/${AWS_BUCKET_NAME}/overfitted/model" -P "models/${AWS_BUCKET_NAME}/overfitted"
RUN wget "http://ghadjeres.s3.amazonaws.com/${AWS_BUCKET_NAME}/early_stopped/model" -P "models/${AWS_BUCKET_NAME}/early_stopped"

# dirty fixes because it's useless to upload the dataset
RUN mkdir -p /root/Data/dataset_cache/PianoMidi \
    && mkdir -p /root/Data/databases/Piano/transcriptions/midi/ \
    && mkdir -p /root/Data/dataset_cache/PianoMidi/PianoMidi-PianoIterator \
    && touch /root/Data/dataset_cache/PianoMidi/PianoMidi-PianoIterator/xbuilt

RUN apt-get install -y gcc-8
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 9
RUN apt-get install -y g++-8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 9
RUN gcc --version
RUN g++ --version
RUN pip install pytorch-fast-transformers

EXPOSE 8080
ENV CUDA_VISIBLE_DEVICES=0

# difference between ARG and ENV
ENV AWS_BUCKET_NAME "${AWS_BUCKET_NAME}"
# PIAv3: uses app_no_region
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cia2", \
            "python", "app_no_region.py", \       
            "--config=models/${AWS_BUCKET_NAME}/config.py", \
            # "-o", \  # not overfitted!
            "--num_workers=0"]