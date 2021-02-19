ARG UBUNTU_VERSION=18.04
ARG ARCH=
ARG CUDA=11.0

FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as nvidia
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=8.0.4.30-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=7.1.3-1
ARG LIBNVINFER_MAJOR_VERSION=7

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        libcublas-${CUDA/./-} \
        cuda-nvrtc-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        curl \
        libcudnn8=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

# Install TensorRT if not building for PowerPC
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

FROM nvidia as python

# Python
# ------

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

ARG PYTHON_VERSION=3.8
ARG PYTHON_MAJOR_VERSION=3
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_MAJOR_VERSION}-pip

RUN python${PYTHON_VERSION} -m pip --no-cache-dir install --upgrade \
    "pip<21.0.1" \
    setuptools

# Some TF tools expect a "python" binary
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}
RUN ln -s $(which python${PYTHON_MAJOR_VERSION}) /usr/local/bin/python

FROM python as common

# TensorFlow
# ----------

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow-gpu
ARG TF_PACKAGE_VERSION=2.4.1
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

# Torch
# -----
ARG TORCH_PACKAGE_VERSION=1.7.1+cu110
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir \
    torch${TORCH_PACKAGE_VERSION:+==${TORCH_PACKAGE_VERSION}} \
    torchvision==0.8.2+cu110 \
    torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# JAX
# ---
ARG JAX_PACKAGE_VERSION=0.2.9
ARG JAXLIB_PACKAGE_VERSION=0.1.61+cuda110
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir \
    jax${JAX_PACKAGE_VERSION:+==${JAX_PACKAGE_VERSION}} \
    jaxlib${JAXLIB_PACKAGE_VERSION:+==${JAXLIB_PACKAGE_VERSION}} -f https://storage.googleapis.com/jax-releases/jax_releases.html

FROM common as base
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

FROM base as jupyter
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir jupyter matplotlib
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0
RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN mkdir /.local && chmod a+rwx /.local
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
EXPOSE 8888

RUN python${PYTHON_VERSION} -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]
