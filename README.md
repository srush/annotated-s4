
* **[Link To The Blog Post](https://srush.github.io/annotated-s4)**


<a href="https://srush.github.io/annotated-s4"><img src="https://user-images.githubusercontent.com/35882/149201164-1723a44a-f34b-467c-94b0-ffda5ebcabbb.png"></a>



## Experiments

#### MNIST Sequence Modeling

```bash
python -m s4.train dataset=mnist layer=s4 train.epochs=100 train.bsz=128 model.d_model=128 model.layer.N=64
```

The following command uses a larger model (5M params) and logs generated samples to wandb every epoch. It achieves 0.36 test NLL (0.52 bits per dimension), a state-of-the-art on this task.
```bash
python -m s4.train dataset=mnist layer=s4 train.epochs=100 train.bsz=50 train.lr=5e-3 train.lr_schedule=true model.d_model=512 model.n_layers=6 model.dropout=0.0 train.weight_decay=0.05 model.prenorm=true model.embedding=true wandb.mode=online train.sample=308 
```

#### QuickDraw Sequence Modeling

```bash
# Default arguments
python -m s4.train dataset=quickdraw layer=s4 train.epochs=10 train.bsz=128 model.d_model=128 model.layer.N=64

# "Run in a day" variant
python -m s4.train dataset=quickdraw layer=s4 train.epochs=1 train.bsz=512 model.d_model=256 model.layer.N=64 model.dropout=0.05
```

#### MNIST Classification

```bash
python -m s4.train dataset=mnist-classification layer=s4 train.epochs=20 train.bsz=128 model.d_model=128 model.dropout=0.25 train.lr=5e-3 train.lr_schedule=true seed=1
```

Gets "best" 99.55% accuracy after 20 epochs @ 17s/epoch on an A100

#### CIFAR-10 Classification

```
python -m s4.train dataset=cifar-classification layer={s4,dss,s4d} train.epochs=100 train.bsz=50 model.n_layers=6 model.d_model=512 model.dropout=0.25 train.lr=5e-3 train.weight_decay=0.01 train.lr_schedule=true seed=1
```

S4 gets "best" 91.23% accuracy after 100 epochs @ 2m16s/epoch on an A100

DSS gets "best" 89.31% accuracy after 100 epochs @ 1m41s/epoch on an A100

S4D gets "best" 89.76% accuracy after 100 epochs @ 1m32s/epoch on an A100

The alternative S4D-Lin initialization performs slightly better with 90.98% accuracy.

```
python -m s4.train dataset=cifar-classification layer=s4d train.epochs=100 train.bsz=50 model.n_layers=6 model.d_model=512 model.dropout=0.25 train.lr=5e-3 train.weight_decay=0.01 train.lr_schedule=true seed=1 +model.layer.scaling=linear
```


---

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

### GPU (CUDA > 11 & CUDNN > 8.2)

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
pip install black celluloid flake8 google-cloud-storage isort ipython matplotlib pre-commit seaborn tensorflow tqdm

# Set up pre-commit
pre-commit install
```

### GPU (CUDA > 11, CUDNN > 8.2)

Note - CUDNN > 8.2 is critical for compilation without warnings, and GPU w/ at least Turing architecture for full
efficiency.

```bash
# Set up virtual/conda environment of your choosing & activate... then install the following:
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install flax
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Defaults
pip install black celluloid flake8 google-cloud-storage isort ipython matplotlib pre-commit seaborn tensorflow tqdm

# Set up pre-commit
pre-commit install
```
