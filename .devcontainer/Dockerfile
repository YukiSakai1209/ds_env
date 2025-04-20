FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=1001
ARG WORK_DIR=/home/${USERNAME}/vermeer/envs/ds_env

# Install basic dependencies and SSHFS
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    build-essential \
    sshfs \
    openssh-client \
    fuse3 \
    wget \
    ca-certificates \
    curl \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev \
    libsdl1.2-dev libsmpeg-dev libportmidi-dev ffmpeg libswscale-dev \
    libavformat-dev libavcodec-dev swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up locale to avoid R/rpy2 warnings
RUN apt-get update && apt-get install -y --no-install-recommends locales \
    && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install essential packages, sudo, git, SSH client, curl, etc.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    build-essential \
    sshfs \
    openssh-client \
    fuse3 \
    wget \
    ca-certificates \
    curl \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev \
    libsdl1.2-dev libsmpeg-dev libportmidi-dev ffmpeg libswscale-dev \
    libavformat-dev libavcodec-dev swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest \
    && node -v \
    && npm -v \
    && npx -v

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN curl -fsSLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x /tmp/miniconda.sh \
    && /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Create the user and necessary directories
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && usermod -aG sudo $USERNAME \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME \
    && mkdir -p /home/$USERNAME/.ssh \
    && mkdir -p ${WORK_DIR} \
    && mkdir -p /home/$USERNAME/.config \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/.ssh \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/vermeer \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/.config \
    && chmod -R 755 /home/$USERNAME/.config \
    && chmod -R 755 /home/$USERNAME/vermeer \
    && chmod -R 755 ${WORK_DIR}

# Add FUSE device access
RUN echo "user_allow_other" >> /etc/fuse.conf

# パッケージリスト更新
RUN apt-get update

# Initialize conda
RUN conda init bash \
    && chown $USERNAME:$USERNAME /home/$USERNAME/.bashrc

# Add environment variables for debugging
ENV DEBUG=true \
    VERBOSE_LOGGING=true

# Initialize shell for conda
SHELL ["/bin/bash", "-c"]

# Copy all environment files
COPY environment.yml /tmp/environment.yml
COPY environment_dynamicviz.yml /tmp/environment_dynamicviz.yml
COPY environment_r.yml /tmp/environment_r.yml
COPY latest_requirements.txt /tmp/latest_requirements.txt

# Create research environment
RUN conda env create -p /opt/conda/envs/research -f /tmp/environment.yml \
    && conda clean -a -y --force
SHELL ["conda", "run", "-n", "research", "/bin/bash", "-c"]
RUN pip install --no-cache-dir --upgrade pip
RUN rm -rf /root/.cache/pip && rm -rf /home/${USERNAME}/.cache/pip
# Install specific wheel URL directly to bypass pip resolution issues
RUN conda run -n research pip install -vvv --no-cache-dir https://download.pytorch.org/whl/cu121/torch-2.3.1%2Bcu121-cp310-cp310-linux_x86_64.whl \
    && rm -rf /root/.cache/pip && rm -rf /home/${USERNAME}/.cache/pip
RUN conda run -n research pip install --no-cache-dir cupy-cuda12x \
    && rm -rf /root/.cache/pip && rm -rf /home/${USERNAME}/.cache/pip
# Upgrade grpcio using pip to satisfy Optuna's requirement (dependency via CuPy -> Optuna)
RUN pip install --no-cache-dir --upgrade "grpcio>=1.68.1"

# Reset shell back to bash
SHELL ["/bin/bash", "-c"]

# Create DynamicViz environment
RUN conda env create -p /opt/conda/envs/dynamicviz -f /tmp/environment_dynamicviz.yml \
    && conda clean -a -y --force

# Create R environment using the updated environment file
COPY environment_r.yml /tmp/environment_r.yml
RUN conda env create -p /opt/conda/envs/r_env -f /tmp/environment_r.yml \
    && conda clean -a -y --force \
    && rm /tmp/environment_r.yml

# Activate env and install R kernel spec system-wide using conda run
RUN conda run -n r_env R -e "IRkernel::installspec(user = FALSE)"

# Install rpy2 using pip as a workaround for conda dependency issues
RUN conda run -n r_env pip install --no-cache-dir rpy2

# Install system dependencies potentially needed by R packages installed via Rscript/devtools
# Kept because pairwiseAdonis/anovakun might need them and aren't installed via conda
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages not available via conda (ensure r-devtools is in environment_r.yml)
# Added CRAN repo option and upgrade='never' for install_github
RUN /opt/conda/envs/r_env/bin/Rscript -e "options(repos = c(CRAN = 'https://cloud.r-project.org')); devtools::install_github('pmartinezarbizu/pairwiseAdonis/pairwiseAdonis', upgrade = 'never')"
COPY anovakun_489.txt /tmp/
RUN /opt/conda/envs/r_env/bin/Rscript -e "source('/tmp/anovakun_489.txt', encoding = 'CP932')"

# TensorFlow GPU env
# RUN conda create -y -p /opt/conda/envs/tf_gpu python=3.11
# # Make RUN commands use the new environment (instead of conda activate):
# SHELL ["conda", "run", "-n", "tf_gpu", "/bin/bash", "-c"]
# RUN pip install --ignore-installed blinker==1.8.2 \
#     && pip install tensorflow-probability==0.25.0 \
#     && pip install tensorflow==2.18.0 \
#     && pip install -U -r /tmp/latest_requirements.txt \
#     && conda clean -a -y

# PyTorch GPU
# RUN conda create -y -p /opt/conda/envs/torch_gpu python=3.11
# # Make RUN commands use the new environment (instead of conda activate):
# SHELL ["conda", "run", "-n", "torch_gpu", "/bin/bash", "-c"]
# RUN conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia \
#     && pip install -U -r /tmp/latest_requirements.txt \
#     && conda clean -a -y

# Clean up
RUN rm -f /tmp/environment*.yml
RUN rm -f /tmp/anovakun_489.txt
RUN rm -f /tmp/latest_requirements.txt

# Reset shell back to bash
SHELL ["/bin/bash", "-c"]

# Initialize shell for conda
RUN conda init bash \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/vscode/.bashrc \
    && echo "conda activate research" >> /home/vscode/.bashrc \
    && echo 'export DEBUG=true' >> /home/vscode/.bashrc \
    && echo 'export VERBOSE_LOGGING=true' >> /home/vscode/.bashrc \
    && echo "if [ -f ${WORK_DIR}/.env ]; then" >> /home/$USERNAME/.bashrc \
    && echo "    source ${WORK_DIR}/.env" >> /home/$USERNAME/.bashrc \
    && echo "fi" >> /home/$USERNAME/.bashrc \
    && chown vscode:vscode /home/vscode/.bashrc        

# Add NVIDIA entrypoint script and CUDA environment variables
RUN mkdir -p /opt/nvidia && \
    echo '#!/bin/bash' > /opt/nvidia/nvidia_entrypoint.sh && \
    echo 'set -e' >> /opt/nvidia/nvidia_entrypoint.sh && \
    echo 'if [[ $# -eq 0 ]]; then' >> /opt/nvidia/nvidia_entrypoint.sh && \
    echo '    exec /bin/bash' >> /opt/nvidia/nvidia_entrypoint.sh && \
    echo 'else' >> /opt/nvidia/nvidia_entrypoint.sh && \
    echo '    exec "$@"' >> /opt/nvidia/nvidia_entrypoint.sh && \
    echo 'fi' >> /opt/nvidia/nvidia_entrypoint.sh && \
    chmod 755 /opt/nvidia/nvidia_entrypoint.sh

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    NVIDIA_VISIBLE_DEVICES=all \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Set default shell to bash
ENV SHELL=/bin/bash

# Make sure the vscode user owns all environments
RUN chown -R vscode:vscode /home/vscode \
    && chown -R vscode:vscode /opt/conda/envs/research \
    && chown -R vscode:vscode /opt/conda/envs/dynamicviz \
    && chown -R vscode:vscode /opt/conda/envs/r_env
    # && chown -R vscode:vscode /opt/conda/envs/tf_gpu
    # && chown -R vscode:vscode /opt/conda/envs/torch_gpu

# Create necessary directories with proper permissions
RUN mkdir -p ${WORK_DIR} \
    && chown -R vscode:vscode /home/vscode \
    && chmod -R 755 /home/vscode \
    && chmod -R 755 /home/vscode/vermeer \
    && chmod -R 755 /home/vscode/vermeer/envs \
    && chmod -R 755 ${WORK_DIR}

# Set default shell to bash
ENV SHELL=/bin/bash

# Copy and set permissions for the entrypoint wrapper
COPY entrypoint_wrapper.sh /usr/local/bin/entrypoint_wrapper.sh
RUN chmod +x /usr/local/bin/entrypoint_wrapper.sh

# Setup ssh config
# Copy SSH config and set ownership directly
COPY --chown=vscode:vscode ssh_config /home/vscode/.ssh/config
# Ensure correct permissions
RUN chmod 600 /home/vscode/.ssh/config

# Switch to vscode user
USER vscode

# Working directory
WORKDIR ${WORK_DIR}
