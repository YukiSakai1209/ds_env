# Neuroscience, Machine Learning, and Data Analysis Environment

This repository contains a computational environment designed for neuroscience research, machine learning applications, and data analysis.

## Libraries and Versions

The following libraries are included in this environment:

- Python: 3.x
- NumPy: x.x.x
- SciPy: x.x.x
- Pandas: x.x.x
- Matplotlib: x.x.x
- Scikit-learn: x.x.x
- Nilearn: x.x.x
- Numba: x.x.x
- Other relevant libraries...
  - Please check the .devcontainer/environment.yml for more information.

Please ensure that you have the correct versions installed for compatibility.


## Logging into GHCR, Pulling the Docker Image, and Testing the Environment in Remote Computing Servers

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
