FROM python:3.7-bullseye

WORKDIR /app/
ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        xvfb && \
    rm -rf /var/lib/apt/lists/*

### ONLY NEEDED FOR SSH 
## Setup user and SSH server
# RUN apt update && apt install  openssh-server sudo -y
# RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 dev 
# RUN  echo 'dev:5595' | chpasswd
# RUN service ssh start
# EXPOSE 22

## install vscode server || 2 options
# 1
# RUN curl -fsSL https://code-server.dev/install.sh | sh
# 2
# COPY setup/download-vs-code-server.sh download-vs-code-server.sh
# RUN bash download-vs-code-server.sh

## install python requirements
COPY setup/requirements.txt tmp/requirements.txt

RUN python3 -m pip install -r tmp/requirements.txt

RUN python3 -m pip install ipykernel -U --user --force-reinstall
# COPY setup/test.py ./test.py
# CMD python3 /app/test.py
### ONLY NEEDED FOR SSH 
# CMD ["/usr/sbin/sshd","-D"]