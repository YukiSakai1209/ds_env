# Neuroscience, Machine Learning, and Data Analysis Environment

This repository contains a computational environment designed for neuroscience research, machine learning applications, and data analysis.

## Conda Environment Setup

This project utilizes the following Conda environments.

### 1. Research Environment

- Environment Name: `research`
- Configuration File: `environment.yml`
- Key Dependencies:
  - Python 3.10
  - NumPy
  - SciPy
  - Pandas
  - Matplotlib
  - Scikit-learn
  - Nilearn
  - Numba
  - Other libraries (see file for details)

### 2. DynamicViz Environment

- Environment Name: `dynamicviz`
- Configuration File: `environment_dynamicviz.yml`
- Key Dependencies:
  - Python 3.8
  - dynamicviz
  - Other libraries (see file for details)

### 3. R Environment

- Environment Name: `r_env`
- Configuration File: `environment_r.yml`
- Key Dependencies:
  - R 4.4.2
  - r-essentials
  - r-vegan
  - r-lme4
  - r-lmerTest
  - Other libraries (see file for details)

This ensures that the necessary libraries are installed in the respective environments.

## Logging into GHCR, Pulling the Docker Image, and Testing the Environment

To use the Docker image, follow these steps:

1. Log in to GHCR:

   ```bash
   echo PAT | docker login ghcr.io -u YukiSakai1209 --password-stdin
   ```

2. Pull the Docker image:

   ```bash
   docker pull ghcr.io/yukisakai1209/ds_env:latest
   ```

3. Test the environment:

   ```bash
   docker run ghcr.io/yukisakai1209/ds_env:latest /opt/conda/envs/research/bin/python -c "import numpy; print('NumPy version:', numpy.__version__)"
   ```
