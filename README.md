# S4

## Quickstart (Development)

We have two `requirements.txt` files that hold dependencies for the current project: one that is tailored to CPUs,
the other that installs for GPU.

### CPU-Only (MacOS, Linux)

```bash 
# Set up virtual/conda environment of your choosing & activate...
pip install -r requirements-cpu.txt
```

### GPU (CUDA 11.3)

```bash
# Set up virtual/conda environment of your choosing & activate...
pip install -r requirements.txt
```

## Dependencies from Scratch

In case the above `requirements.txt` don't work, here are the commands used to download dependencies.

### CPU-Only

```bash
# Set up virtual/conda environment of your choosing & activate... then install the following:
pip install --upgrade "jax[cpu]"
pip install torch torchvision torchaudio
pip install flax

# Defaults
pip install ipython matplotlib
```

### GPU (CUDA > 11, CUDNN > 8.0.5)

```bash
# Set up virtual/conda environment of your choosing & activate... then install the following:
pip install jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install torch torchvision torchaudio
pip install flax

# Defaults
pip install ipython matplotlib 
```
