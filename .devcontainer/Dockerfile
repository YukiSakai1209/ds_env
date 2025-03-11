FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=1001

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
    && rm -rf /var/lib/apt/lists/*

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
    && mkdir -p /home/$USERNAME/vermeer/envs/ds_env \
    && mkdir -p /home/$USERNAME/.config/Windsurf/logs \
    && mkdir -p /home/$USERNAME/.windsurf \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/.ssh \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/vermeer \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/.config \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/.windsurf \
    && chmod -R 755 /home/$USERNAME/.config/Windsurf/logs \
    && chmod -R 755 /home/$USERNAME/vermeer \
    && chmod -R 755 /home/$USERNAME/vermeer/envs/ds_env

# Add FUSE device access
RUN echo "user_allow_other" >> /etc/fuse.conf

# パッケージリスト更新
RUN apt-get update

# Initialize conda
RUN conda init bash \
    && chown $USERNAME:$USERNAME /home/$USERNAME/.bashrc

# Add environment variables for debugging
ENV DEBUG=true \
    VERBOSE_LOGGING=true \
    WINDSURF_DEBUG=true \
    WINDSURF_LOG_LEVEL=trace \
    WINDSURF_EXTENSION_LOG_LEVEL=trace

# Initialize shell for conda
SHELL ["/bin/bash", "-c"]

# Copy all environment files
COPY environment*.yml /tmp/
COPY latest_requirements.txt /tmp/

# Create research environment
RUN conda env create -p /opt/conda/envs/research -f /tmp/environment.yml \
    && conda clean -a -y

# Create DynamicViz environment
RUN conda env create -p /opt/conda/envs/dynamicviz -f /tmp/environment_dynamicviz.yml \
    && conda clean -a -y

# Create R environment
RUN conda env create -p /opt/conda/envs/r_env -f /tmp/environment_r.yml \
    && conda clean -a -y

# Install R packages in r_env
RUN /opt/conda/envs/r_env/bin/Rscript -e "devtools::install_github('pmartinezarbizu/pairwiseAdonis/pairwiseAdonis')"
COPY anovakun_489.txt /tmp/
RUN /opt/conda/envs/r_env/bin/Rscript -e "source('/tmp/anovakun_489.txt', encoding = 'CP932')"

# TensorFlow GPU env
RUN conda create -y -p /opt/conda/envs/tf_gpu python=3.11
# Make RUN commands use the new environment (instead of conda activate):
SHELL ["conda", "run", "-n", "tf_gpu", "/bin/bash", "-c"]
RUN pip install --ignore-installed blinker==1.8.2 \
    && pip install tensorflow-probability==0.25.0 \
    && pip install tensorflow==2.18.0 \
    && pip install -U -r /tmp/latest_requirements.txt \
    && conda clean -a -y

# PyTorch GPU
RUN conda create -y -p /opt/conda/envs/torch_gpu python=3.11
# Make RUN commands use the new environment (instead of conda activate):
SHELL ["conda", "run", "-n", "torch_gpu", "/bin/bash", "-c"]
RUN conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia \
    && pip install -U -r /tmp/latest_requirements.txt \
    && conda clean -a -y

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
    && echo 'export WINDSURF_DEBUG=true' >> /home/vscode/.bashrc \
    && echo 'export WINDSURF_LOG_LEVEL=trace' >> /home/vscode/.bashrc \
    && echo 'export WINDSURF_EXTENSION_LOG_LEVEL=trace' >> /home/vscode/.bashrc \
    && echo 'if [ -f /home/vscode/vermeer/envs/ds_env/.env ]; then' >> /home/$USERNAME/.bashrc \
    && echo '    source /home/vscode/vermeer/envs/ds_env/.env' >> /home/$USERNAME/.bashrc \
    && echo 'fi' >> /home/$USERNAME/.bashrc \
    && echo 'export NVIDIA_ENTRYPOINT=/home/vscode/vermeer/envs/ds_env/entrypoint.sh' >> /home/$USERNAME/.bashrc \
    && echo 'if [ -f "$NVIDIA_ENTRYPOINT" ]; then' >> /home/$USERNAME/.bashrc \
    && echo '    source "$NVIDIA_ENTRYPOINT"' >> /home/$USERNAME/.bashrc \
    && echo 'fi' >> /home/$USERNAME/.bashrc \
    && chown vscode:vscode /home/vscode/.bashrc

# NVIDIAエントリポイントスクリプトとCUDA環境変数の追加
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
    && chown -R vscode:vscode /opt/conda/envs/r_env \
    && chown -R vscode:vscode /opt/conda/envs/tf_gpu \
    && chown -R vscode:vscode /opt/conda/envs/torch_gpu

# Create necessary directories with proper permissions
RUN mkdir -p /home/vscode/vermeer/envs/ds_env \
    && chown -R vscode:vscode /home/vscode/vermeer \
    && chmod -R 755 /home/vscode/vermeer \
    && chmod -R 755 /home/vscode/vermeer/envs \
    && chmod -R 755 /home/vscode/vermeer/envs/ds_env

# Set default shell to bash
ENV SHELL=/bin/bash

# Switch to vscode user
USER vscode

# Working directory
WORKDIR /home/vscode/vermeer/envs/ds_env
