# Revisit-Anything (SegVLAD) build image.
# Modernized to mirror VLAD-BuFF/environment.yml: Python 3.10, torch 2.1.0+cu121,
# pytorch-lightning 2.1.2. Required for Ada-generation GPUs (sm_89, e.g. L4),
# which have no kernels in torch 1.11/cu113. The upstream segvlad.yaml (py38,
# cuda10.2) is not reusable as-is and is superseded by the in-tree VLAD-BuFF
# config.
#
# Build from the repo root:
#   docker build -t revisit-anything .
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PATH=/opt/conda/bin:$PATH \
  CONDA_DIR=/opt/conda \
  PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  wget \
  git \
  bzip2 \
  build-essential \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
  && rm /tmp/miniconda.sh \
  && conda clean -afy

WORKDIR /workspace/Revisit-Anything

# Conda env: Python 3.10 + torch 2.1.0 (cu121) + faiss-gpu from pytorch channel.
# Including `pip` explicitly so subsequent pip installs target the segvlad env,
# not base conda. --override-channels avoids pkgs/main / pkgs/r ToS prompts.
RUN conda create -y -n segvlad --override-channels \
  -c pytorch -c nvidia -c conda-forge \
  python=3.10 \
  pip \
  pytorch=2.1.0 \
  pytorch-cuda=12.1 \
  torchvision=0.16.0 \
  faiss-gpu=1.9.0 \
  && conda clean -afy

# Activate segvlad via PATH for all subsequent RUN steps and at runtime.
# This is more reliable than `SHELL ["conda", "run", ...]`, which can dispatch
# to base-conda's pip if the env is missing pip.
ENV CONDA_DEFAULT_ENV=segvlad \
  PATH=${CONDA_DIR}/envs/segvlad/bin:${PATH}

# Pip deps: lightning + the project's runtime requirements.
# Versions for lightning / metric-learning / torchmetrics mirror
# VLAD-BuFF/environment.yml so the container matches a config the upstream
# subproject already runs against. `python -m pip` to guarantee segvlad's pip.
RUN python -m pip install \
  "setuptools<70" \
  pytorch-lightning==2.1.2 \
  pytorch-metric-learning==2.3.0 \
  torchmetrics==1.2.0 \
  einops \
  fast-pytorch-kmeans \
  h5py \
  huggingface-hub \
  matplotlib \
  natsort \
  "numpy<2" \
  opencv-python \
  pandas \
  pillow \
  pynvml \
  pyyaml \
  scikit-learn \
  scipy \
  tqdm \
  transformers==4.36.2 \
  typer \
  tyro \
  utm \
  wandb \
  psutil \
  prettytable \
  hydra-core \
  loguru


COPY . ./
RUN python -m pip install -e ./sam

# Datasets / model checkpoints are mounted at runtime; see
# README "Dataset preparation" for the expected workdir_data layout.
CMD ["bash"]
