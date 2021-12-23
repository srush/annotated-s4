# S4

## Quickstart (Development)

We have two `requirements.txt` files that hold dependencies for the current project: one that is tailored to CPUs,
the other that installs for GPU.

### CPU-Only (MacOS, Linux)

```bash
# Set up virtual/conda environment of your choosing & activate...
pip install -r requirements-cpu.txt

# Set up pre-commit
pre-commit install
```

### GPU (CUDA > 11 & CUDNN > 8.0.5)

```bash
# Set up virtual/conda environment of your choosing & activate...
pip install -r requirements-gpu.txt

# Set up pre-commit
pre-commit install
```

## Dependencies from Scratch

In case the above `requirements.txt` don't work, here are the commands used to download dependencies.

### CPU-Only

```bash
# Set up virtual/conda environment of your choosing & activate... then install the following:
pip install --upgrade "jax[cpu]"
pip install flax
pip install torch torchvision torchaudio

# Defaults
pip install black flake8 isort ipython matplotlib pre-commit tqdm

# Set up pre-commit
pre-commit install
```

### GPU (CUDA > 11, CUDNN > 8.0.5)

```bash
# Set up virtual/conda environment of your choosing & activate... then install the following:
pip install jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install flax
pip install torch torchvision torchaudio

# Defaults
pip install black flake8 isort ipython matplotlib pre-commit tqdm

# Set up pre-commit
pre-commit install
```
