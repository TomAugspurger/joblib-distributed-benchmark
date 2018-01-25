FROM ubuntu:16.04 as builder
MAINTAINER Martin Durant <martin.durant@utoronto.ca>

RUN apt-get update -yqq \
    && apt-get install -yqq build-essential bzip2 git wget graphviz \
    && rm -rf /var/lib/apt/lists/*

# Configure environment
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH="/work/bin:/work/miniconda/bin:$PATH"

RUN mkdir -p /work/bin

# Add local files at the end of the Dockerfile to limit cache busting
# COPY config /work/config
COPY config /work/config

# Install Python 3 from miniconda
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh

# Install pydata stack
RUN bash miniconda.sh -b -p /work/miniconda && rm miniconda.sh \
 && conda config --set always_yes yes --set changeps1 no --set auto_update_conda no \
 && conda install -c conda-forge \
    notebook \
    psutil \
    numpy \
    pandas \
    pip \
    nomkl \
    lz4 \
    tornado \
    python-blosc \
    cytoolz \
    dask \
    distributed \
    jupyterlab \
    bokeh \
    nodejs \
    Cython \
    && conda clean -tipsy
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN pip install git+https://github.com/dask/dask --upgrade --no-deps
RUN pip install git+https://github.com/dask/distributed --upgrade --no-deps

RUN git clone https://github.com/TomAugspurger/joblib -b ogrisel-dev-shm-size \
 && cd joblib \
 && pip install -e .

RUN git clone https://github.com/TomAugspurger/scikit-learn -b joblib-hints \
 && cd scikit-learn \
 && python setup.py build_ext -i -j 4 \
 && pip install -e . \
 && cd sklearn/externals \
 && ./copy_joblib.sh ../../../joblib

RUN conda clean -tipsy

FROM ubuntu:16.04

ENV BASICUSER basicuser
ENV BASICUSER_UID 1000
RUN useradd -m -d /work -s /bin/bash -N -u $BASICUSER_UID $BASICUSER

COPY --from=builder --chown=basicuser:users ./work work

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH="/work/bin:/work/miniconda/bin:$PATH"

# Install Tini that necessary to properly run the notebook service in docker
# http://jupyter-notebook.readthedocs.org/en/latest/public_server.html#docker-cmd
ENV TINI_VERSION v0.9.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# for further interaction with kubernetes

ADD https://storage.googleapis.com/kubernetes-release/release/v1.5.4/bin/linux/amd64/kubectl /usr/sbin/kubectl
RUN chmod +x /usr/bin/tini && chmod 0500 /usr/sbin/kubectl

# COPY examples /work/examples
ENTRYPOINT ["/usr/bin/tini", "--"]
