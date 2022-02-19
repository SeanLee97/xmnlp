FROM ubuntu:18.04

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8 python3.8-venv \
    && python3.8 -m ensurepip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

ADD dev-requirements.txt /tmp/dev-requirements.txt
ADD requirements.txt /tmp/requirements.txt
ADD xmnlp/xmnlp-onnx-models /home/xmnlp/xmnlp-onnx-models

ARG PIP_INDEX_URL
RUN python3.8 -m pip install --no-cache-dir -U pip twine flake8 && \
    python3.8 -m pip install --no-cache-dir -r /tmp/dev-requirements.txt
