FROM python:3.10.12-bullseye AS builder
# Stage 1: Builder/Compiler
# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc gnupg dirmngr

# Adding GPG keys
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 04EE7237B7D453EC && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138

# Adding repository configurations
RUN echo "deb http://deb.debian.org/debian bullseye main" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bullseye-security main" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bullseye-updates main" >> /etc/apt/sources.list

# Updating package list again after adding repositories
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc


WORKDIR /app
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

COPY project/requirements.txt requirements.txt
RUN python -m pip install -U pip setuptools
RUN python -m pip install -r requirements.txt

# Stage 2: Runtime
# FROM nvidia/cuda:10.1-cudnn7-runtime
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

WORKDIR /app
RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.10 libmagic-dev poppler-utils tesseract-ocr python3-distutils && \
    apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN apt update && \
    apt -y install curl && \
    curl -fsSL https://ollama.com/install.sh | sh

COPY --from=builder /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN ln -sf /usr/bin/python3 /app/venv/bin/python

ENV PYTHONPATH=/app
EXPOSE 8000

COPY . .

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]