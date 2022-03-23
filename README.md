
* **[Link To The Blog Post](https://srush.github.io/annotated-s4)**


<a href="https://srush.github.io/annotated-s4"><img src="https://user-images.githubusercontent.com/35882/149201164-1723a44a-f34b-467c-94b0-ffda5ebcabbb.png"></a>



## Experiments

#### MNIST Sequence Modeling

```bash
# Default arguments
python -m s4.train --dataset mnist --model s4 --epochs 100 --bsz 128 --d_model 128 --ssm_n 64
```

#### QuickDraw Sequence Modeling

```bash
# Default arguments
python -m s4.train --dataset quickdraw --model s4 --epochs 10 --bsz 128 --d_model 128 --ssm_n 64

# "Run in a day" variant
python -m s4.train --dataset quickdraw --model s4 --epochs 1 --bsz 512 --d_model 256 --ssm_n 64 --p_dropout 0.05
```

#### MNIST Classification

```bash
# Default arguments
python -m s4.train --dataset mnist-classification --model s4 --epochs 10 --bsz 128 --d_model 128 --ssm_n 64
```

(Default Arguments, as shown above): Gets "best" 97.76% accuracy in 10 epochs @ 40s/epoch on a TitanRTX.

#### CIFAR-10 Classification

```bash
## Adding a Cubic Decay Schedule for last 70% of training

# Default arguments (100 epochs for CIFAR)
python -m s4.train --dataset cifar-classification --model s4 --epochs 100 --bsz 128 --d_model 128 --ssm_n 64 --lr 1e-2 --lr_schedule

# S4 replication from central repository
python -m s4.train --dataset cifar-classification --model s4 --epochs 100 --bsz 64 --d_model 512 --ssm_n 64 --lr 1e-2 --lr_schedule

## After Fixing S4-Custom Optimization & Dropout2D (all implemented inline now... can add flags if desired)

# Default arguments (100 epochs for CIFAR)
python -m s4.train --dataset cifar-classification --model s4 --epochs 100 --bsz 128 --d_model 128 --ssm_n 64 --lr 1e-2

# S4 replication from central repository
python -m s4.train --dataset cifar-classification --model s4 --epochs 100 --bsz 64 --d_model 512 --ssm_n 64 --lr 1e-2

## Before Fixing S4-Custom Optimization...

# Default arguments (100 epochs for CIFAR)
python -m s4.train --dataset cifar-classification --model s4 --epochs 100 --bsz 128 --d_model 128 --ssm_n 64

# S4 replication from central repository
python -m s4.train --dataset cifar-classification --model s4 --epochs 100 --bsz 64 --d_model 512 --ssm_n 64
```

Adding a Schedule:
- (LR 1e-2 w/ Replication Args -- "big" model): 71.55% (still running, 39 epochs) @ 3m16s on a TitanRTX
- (LR 1e-2 w/ Default Args -- not "bigger" model): 71.92% @ 36s/epoch on a TitanRTX

After Fixing Dropout2D (w/ Optimization in Place):
- (LR 1e-2 w/ Replication Args -- "big" model): 70.68% (still running, 47 epochs) @ 3m17s on a TitanRTX
- (LR 1e-2 w/ Default Args -- not "bigger" model): 68.20% @ 36s/epoch on a TitanRTX

After Fixing Optimization, Before Fixing Dropout2D:
- (LR 1e-2 w/ Default Args -- not "bigger" model): 67.14% @ 36s/epoch on a TitanRTX 

Before Fixing S4 Optimization -- AdamW w/ LR 1e-3 for ALL Parameters:

- (Default Arguments): Gets "best" 63.51% accuracy @ 46s/epoch on a TitanRTX
- (S4 Arguments): Gets "best" 66.44% accuracy @ 3m11s on a TitanRTX
    + Possible reasons for failure to meet replication: LR Schedule (Decay on Plateau), Custom LR per Parameter.
    
**CIFAR Results -- v2**

```
# Following @frederick0329's comment: https://github.com/srush/annotated-s4/pull/43#issuecomment-1065444261
python -m s4.train --dataset cifar-classification --model s4 --epoch 100 --bsz 64 --n_layers 6 --p_dropout 0.25 --lr 1e-2 --d_model 1024

# After @albertfgu's follow-up: https://github.com/srush/annotated-s4/pull/43#issuecomment-1067046738
python -m s4.train --dataset cifar-classification --model s4 --epoch 100 --bsz 64 --n_layers 6 --p_dropout 0.25 --lr 5e-3 --d_model 512

# `n_layers` seems to help, bumping to 8?
python -m s4.train --dataset cifar-classification --model s4 --epoch 100 --bsz 1 --n_layers 8 --p_dropout 0.25 --lr 5e-3 --d_model 512
```

V2 with @frederick0329's comment:
- (`n_layers=6, d_model=1024`) Gets "best" 84.12% accuracy after 100 epochs @ 7m3s/epoch on a TitanRTX

V2 w/ @albertfgu's advice:
- (`n_layers=6, d_model=512, lr=5e-3`) Gets "best" 85.81% accuracy after 100 epochs @ 3m8s/epoch on a TitanRTX

V2 w/ `n_layers=8`:
- (`n_layers=8, d_model=512, lr=5e-3`) Gets "best" 86.35% accuracy after 100 epochs @ 3m41s/epoch on a TitanRTX


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
