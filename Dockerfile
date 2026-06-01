FROM python:3.12-slim

ARG WORKSPACE_DIR=/workspace/hsp
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
  && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

ENV PYTHONUNBUFFERED=1
WORKDIR ${WORKSPACE_DIR}
VOLUME ["${WORKSPACE_DIR}"]
USER ${USERNAME}

CMD ["bash"]
