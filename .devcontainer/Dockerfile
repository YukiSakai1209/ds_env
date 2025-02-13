FROM mcr.microsoft.com/devcontainers/miniconda:3

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
    && rm -rf /var/lib/apt/lists/*

# Create the user and necessary directories
RUN usermod -u 1001 vscode \
    && groupmod -g 1001 vscode \
    && chown -R vscode:vscode /home/vscode \
    && mkdir -p /home/vscode/.ssh \
    && mkdir -p /home/vscode/vermeer/container_envs \
    && mkdir -p /home/vscode/vermeer/ds_env \
    && mkdir -p /home/vscode/.config/Windsurf/logs \
    && mkdir -p /home/vscode/.windsurf \
    && chown -R vscode:vscode /home/vscode/.ssh \
    && chown -R vscode:vscode /home/vscode/vermeer \
    && chown -R vscode:vscode /home/vscode/.config \
    && chown -R vscode:vscode /home/vscode/.windsurf \
    && chmod -R 755 /home/vscode/.config/Windsurf/logs

# Add FUSE device access
RUN echo "user_allow_other" >> /etc/fuse.conf

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

# Clean up
RUN rm -f /tmp/environment*.yml
RUN rm -f /tmp/anovakun_489.txt

# Initialize shell for conda
RUN conda init bash \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/vscode/.bashrc \
    && echo "conda activate research" >> /home/vscode/.bashrc \
    && echo 'export DEBUG=true' >> /home/vscode/.bashrc \
    && echo 'export VERBOSE_LOGGING=true' >> /home/vscode/.bashrc \
    && echo 'export WINDSURF_DEBUG=true' >> /home/vscode/.bashrc \
    && echo 'export WINDSURF_LOG_LEVEL=trace' >> /home/vscode/.bashrc \
    && echo 'export WINDSURF_EXTENSION_LOG_LEVEL=trace' >> /home/vscode/.bashrc \
    && chown vscode:vscode /home/vscode/.bashrc

# Set default shell to bash
ENV SHELL=/bin/bash

# Make sure the vscode user owns all environments
RUN chown -R vscode:vscode /home/vscode \
    && chown -R vscode:vscode /opt/conda/envs/research \
    && chown -R vscode:vscode /opt/conda/envs/dynamicviz \
    && chown -R vscode:vscode /opt/conda/envs/r_env

USER $USERNAME
