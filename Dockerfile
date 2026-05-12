# Revisit-Anything (SegVLAD) + DetectorFreeSfM combined build image.
# Base: CUDA 12.1 / cuDNN 8 (devel) on Ubuntu 22.04. CUDA 12.1 is required for
# Ada-generation GPUs (sm_89, e.g. L4), which have no kernels in torch 1.11/cu113.
# Python 3.10 + torch 2.1.0+cu121 lives in the `segvlad` conda env and is shared
# by both Revisit-Anything and the DetectorFreeSfM submodule.
#
# SphereSfM / pycolmap are vendored as git submodules under submodules/. Make
# sure to `git submodule update --init --recursive` before building.
#
# Build:
#   docker build -t revisit-anything .
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PATH=/opt/conda/bin:$PATH \
  CONDA_DIR=/opt/conda \
  PIP_NO_CACHE_DIR=1 \
  QT_XCB_GL_INTEGRATION=xcb_egl

# Base + SfM build deps (Ceres, COLMAP, Qt5, Boost, CGAL, SuiteSparse, ...).
RUN apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  wget \
  git \
  bzip2 \
  build-essential \
  cmake \
  ninja-build \
  libgl1 \
  libglib2.0-0 \
  libboost-program-options-dev \
  libboost-filesystem-dev \
  libboost-graph-dev \
  libboost-system-dev \
  libboost-test-dev \
  libeigen3-dev \
  libflann-dev \
  libfreeimage-dev \
  libmetis-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libsqlite3-dev \
  libglew-dev \
  qtbase5-dev \
  libqt5opengl5-dev \
  libcgal-dev \
  libopengl-dev \
  libcgal-qt5-dev \
  libatlas-base-dev \
  libsuitesparse-dev \
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

# Pip deps for Revisit-Anything (VLAD-BuFF) runtime. Versions for lightning /
# metric-learning / torchmetrics mirror VLAD-BuFF/environment.yml.
# `python -m pip` guarantees segvlad's pip.
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
  seaborn \
  tqdm \
  transformers==4.36.2 \
  typer \
  tyro \
  utm \
  wandb \
  psutil \
  py-cpuinfo \
  prettytable \
  hydra-core \
  loguru \
  boto3 \
  python-dotenv \
  ultralytics==8.0.120

# Pip deps for DetectorFreeSfM (shares the segvlad env with Revisit-Anything).
# Mirrors submodules/DetectorFreeSfM/requirements.txt + pyproject.toml extras,
# excluding torch (already provided by conda) and pickle5 (built into Py3.10).
RUN python -m pip install \
  kornia \
  kornia-rs \
  yacs \
  omegaconf \
  tenacity \
  joblib \
  ray==2.7.1 \
  timm \
  albumentations \
  rich \
  poetry \
  poethepoet

# Ceres Solver 2.0.0 (CUDA-enabled). CUDA_ARCH_BIN matches the target GPU
# (sm_89 = L4). Override at build with --build-arg CUDA_ARCH_BIN=...
ARG CERES_VERSION=2.0.0
ARG CUDA_ARCH_BIN=8.9
ARG CUDA_ARCHITECTURES=89
WORKDIR /usr/src/ceres-solver
RUN wget http://ceres-solver.org/ceres-solver-${CERES_VERSION}.tar.gz \
  && tar -xzf ceres-solver-${CERES_VERSION}.tar.gz \
  && mv ceres-solver-${CERES_VERSION} ceres-solver-src \
  && rm ceres-solver-${CERES_VERSION}.tar.gz \
  && mkdir -p ceres-solver-src/build
WORKDIR /usr/src/ceres-solver/ceres-solver-src/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DCERES_BUILD_TESTS=OFF \
    -DCERES_BUILD_EXAMPLES=OFF \
    -DCERES_USE_CUDA=ON \
    -DCERES_USE_CUBLAS=ON \
    -DCERES_USE_CUDNN=ON \
    -DBUILD_DOCS=OFF \
    -DBUILD_PYTHON_MODULE=OFF \
    -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
  && make -j$(nproc) \
  && make install \
  && ldconfig

# SphereSfM + pycolmap (SoftRoid-Inc forks, vendored as submodules). Copied
# independently from the rest of the source so that edits in Revisit-Anything
# don't invalidate these expensive build layers.
COPY submodules/SphereSfM /code/SphereSfM
WORKDIR /code/SphereSfM
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    make -j$(nproc) && make install

COPY submodules/pycolmap /code/pycolmap
WORKDIR /code/pycolmap
RUN python -m pip install .

# Source tree (Revisit-Anything + submodules/DetectorFreeSfM)
WORKDIR /workspace/Revisit-Anything
COPY . ./
RUN python -m pip install -e ./sam

# FastSAM (vendored as submodule). The `fastsam` package is on PYTHONPATH;
# ultralytics 8.0.120 (matching the version FastSAM was built against) is
# installed from pip above.
ENV PYTHONPATH=/workspace/Revisit-Anything/submodules/FastSAM:${PYTHONPATH}

# DetectorFreeSfM uses poethepoet tasks defined in its pyproject.toml; we install
# poetry-managed packages directly into the segvlad env (no nested venv).
RUN poetry config virtualenvs.create false

# Datasets / model checkpoints are mounted at runtime; see
# README "Dataset preparation" for the expected workdir_data layout.
CMD ["bash"]
