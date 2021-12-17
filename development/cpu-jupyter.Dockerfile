ARG UBUNTU_VERSION=18.04
FROM ubuntu:${UBUNTU_VERSION} as base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl

FROM base as python

# Python
# ------

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

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
ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=
RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

# Torch
# -----
# ARG TORCH_PACKAGE_VERSION=1.7.1+cu110
# RUN python${PYTHON_VERSION} -m pip install --no-cache-dir \
#     torch${TORCH_PACKAGE_VERSION:+==${TORCH_PACKAGE_VERSION}} \
#     torchvision==0.8.2+cu110 \
#     torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# JAX
# ---
# ARG JAX_PACKAGE_VERSION=0.2.9
# ARG JAXLIB_PACKAGE_VERSION=0.1.61+cuda110
# RUN python${PYTHON_VERSION} -m pip install --no-cache-dir \
#     jax${JAX_PACKAGE_VERSION:+==${JAX_PACKAGE_VERSION}} \
#     jaxlib${JAXLIB_PACKAGE_VERSION:+==${JAXLIB_PACKAGE_VERSION}} -f https://storage.googleapis.com/jax-releases/jax_releases.html

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get update && apt-get install -y --no-install-recommends \
        msttcorefonts \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-xetex \
        cm-super \
        dvipng \
        pandoc \
        imagemagick \
        ffmpeg \
        graphviz \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* ~/.cache/matplotlib

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

FROM common as jupyter
RUN python3 -m pip install --no-cache-dir jupyter matplotlib
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
# Pin jedi; see https://github.com/ipython/ipython/issues/12740
RUN python3 -m pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0 jedi==0.17.2
RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN mkdir /.local && chmod a+rwx /.local
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
EXPOSE 8888

RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]
