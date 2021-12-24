# # S4
#
# <h4>
#   <a href="https://arxiv.org/abs/2111.00396" target="_blank">
#       Efficiently Modeling Long Sequences with Structured State Spaces
#   </a>
# </h4>
#
# The recent Structured State Space for Sequence Modeling (S4) architecture has been applied to several difficult
# sequence modeling tasks, showing a remarkable capacity for reasoning over long-term dependencies.
#
# In this post, we...


# # Table of Contents
#
# We're going to develop S4 from first principles – at first, starting with the fundamental state space model,
# showing how optimizing the model naively is difficult. We then step through each of the optimizations and insights
# in the paper, showing how we can scale S4 across time and input dimension.
#
# Here's a brief roadmap:
# - Part I: Developing Intuition for the Fundamental State Space Model
# - Part II: Optimizing S4 for Training & Inference
# - Part III: Putting S4 to the Test


# # Part 0: Preliminaries

# We'll be using Jax to build S4 (see notes at the end for justification)
from functools import partial
import jax
import jax.numpy as np
import numpy as onp
import optax
import torch
import torchvision
import torchvision.transforms as transforms
from flax import linen as nn
from flax.training import train_state
from jax.numpy.linalg import eig, inv
from jax.numpy.linalg import matrix_power as power
from jax.scipy.signal import convolve
from torch.utils.data import TensorDataset
from tqdm import tqdm


# ## Simple Sequence Modeling Datasets
# To show how S4 behaves on various sequence modeling tasks, we create three simple datasets, ranging from a simple toy
# overfitting example, to a more complex $sin$ function tracing problem, finally ending with an MNIST image generation
# task.


# ### $sin(x)$
# **Task**: Overfit to a 8-bit quantized sin(x) from 0 - 2*Pi -- sampled 360 times.
#
#  @Note: The Feed-Forward model won't necessarily be able to fit this data (optimization is hard)
#  As a sanity check, you can try running with N_CLASSES = 2 (-1, 1) and d_model = 1...
#  this is the simplest "majority rule" experiment => gets 100% test accuracy.
#
#  @Note: RNN & S4 *should* fit this perfectly... but needs to be verified.


def create_sin_x_dataset(n_examples=1024, bsz=128):
    print("[*] Generating Toy Dataset: sin(x)...")

    # Constants
    SEQ_LENGTH, N_CLASSES = 360, 8
    x = onp.linspace(0, 2 * onp.pi, num=SEQ_LENGTH)
    y = onp.digitize(onp.sin(x), onp.linspace(-1, 1, num=N_CLASSES))

    # Tile this `n_examples` times...
    data = torch.Tensor(
        onp.tile(onp.expand_dims(onp.expand_dims(y, -1), 0), reps=[n_examples, 1, 1])
    )

    # Build Datasets -- Two entries to match (inputs, targets) structure
    train = TensorDataset(data, data)
    test = TensorDataset(data[:1], data[:1])

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(train, batch_size=bsz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=bsz, shuffle=False)

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH


# ### $sin(ax + b)$
# **Task**: Fit arbitrary 8-bit quantized functions of the form sin(ax + b) from 0 - 2*Pi -- sampled 360 times.
#
# In this dataset, `a` controls amplitude and `b` controls phase and are sampled uniformly at random in prespecified
# intervals.


def create_sin_ax_b_dataset(n_examples=20000, bsz=128):
    print("[*] Generating sin(ax + b) Dataset...")

    # Constants – `a` sampled uniform from [1, 10], `b` sampled uniform [0, 5]
    SEQ_LENGTH, N_CLASSES, A_MAX, B_MAX = 360, 8, 10, 5
    train_data, test_data = [], []
    data_key = jax.random.PRNGKey(21)

    # Loop through `n_examples` and generate data
    print(f"\t=>> Generating {n_examples} Training Examples...")
    x = onp.linspace(0, 2 * onp.pi, num=SEQ_LENGTH)
    for _ in tqdm(range(n_examples)):
        data_key, a_rng, b_rng = jax.random.split(data_key, num=3)

        # Compute a, b
        a, b = jax.random.uniform(a_rng, minval=1.0, maxval=A_MAX), jax.random.uniform(
            b_rng, maxval=B_MAX
        )
        train_data.append(
            onp.digitize(onp.sin(a * x + b), onp.linspace(-1, 1, num=N_CLASSES))
        )

    # Generate 1 Batch of Test Examples
    print(f"\t=>> Generating {bsz} Test Examples...")
    for _ in tqdm(range(bsz)):
        data_key, a_rng, b_rng = jax.random.split(data_key, num=3)

        # Compute a, b
        a, b = jax.random.uniform(a_rng, minval=1.0, maxval=A_MAX), jax.random.uniform(
            b_rng, maxval=B_MAX
        )
        test_data.append(
            onp.digitize(onp.sin(a * x + b), onp.linspace(-1, 1, num=N_CLASSES))
        )

        # Build Datasets - Two entries to match (inputs, targets) structure
        train_data = torch.Tensor(onp.expand_dims(onp.array(train_data), -1))
        test_data = torch.Tensor(onp.expand_dims(onp.array(test_data), -1))
        train = TensorDataset(train_data, train_data)
        test = TensorDataset(test_data, test_data)

        # Return data loaders, with the provided batch size
        trainloader = torch.utils.data.DataLoader(train, batch_size=bsz, shuffle=True)
        testloader = torch.utils.data.DataLoader(test, batch_size=bsz, shuffle=False)

        return trainloader, testloader, N_CLASSES, SEQ_LENGTH


# ### MNIST Sequence Modeling
# **Task**: Predict next pixel value given history, in an autoregressive fashion (784 pixels x 256 values).
#
# While we train on full sequences, generations should probably condition on first 10-25% of image.


def create_mnist_dataset(bsz=128):
    print("[*] Generating MNIST Sequence Modeling Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES = 784, 256

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(1, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(train, batch_size=bsz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=bsz, shuffle=False)

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH


# ## Baseline Models
#
# We start with definitions of various models we're already familiar with, starting with a feed-forward
# (history-blind) projection model, followed by a strong LSTM-based recurrent baseline.

# ### Utilities
# We define a couple of utility functions below to compute a standard cross-entropy loss, and compute
# "token"-level prediction accuracy.


def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels[..., 0], num_classes=logits.shape[-1])
    return -np.mean(np.sum(one_hot_labels * logits, axis=-1))


def compute_accuracy(logits, labels):
    return np.mean(np.argmax(logits, -1) == labels.squeeze())


# As we're using Flax, we also write a utility function to return a default TrainState object.
# This function initializes model parameters, as well as our optimizer.


def create_train_state(model, init_rng, dropout_rng, bsz=128, seq_len=784, lr=1e-3):
    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        np.ones((bsz, seq_len - 1, 1)),
    )["params"]
    tx = optax.adamw(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# We also use this opportunity to write generic train_epoch and validation functions. These functions generally
# operate by taking in a training state, model class, dataloader, and critically, the model-specific step function.
# We define the step functions on a model-specific basis below.


def train_epoch(state, model, trainloader, train_step):
    # Store Metrics
    batch_losses = []
    for batch_idx, (inputs, _) in enumerate(tqdm(trainloader)):
        inputs = np.array(inputs.numpy())
        state, loss = train_step(state, inputs, model)
        batch_losses.append(loss)

    # Return average loss over batches
    return np.mean(np.array(batch_losses))


def validate(params, model, testloader, eval_step):
    # Compute average loss & accuracy
    losses, accuracies = [], []
    for batch_idx, (inputs, _) in enumerate(tqdm(testloader)):
        inputs = np.array(inputs.numpy())
        loss, acc = eval_step(inputs, params, model)
        losses.append(loss)
        accuracies.append(acc)

    # Sampling autoregressively prompted w/ first 100 "tokens"...
    #   => TODO @Sidd
    return np.mean(np.array(losses)), np.mean(np.array(accuracies))


# ### Feed-Forward Model
# Here, we establish a skeleton for a simple, history-blind feed-forward model. For each element $x_t$ of a sequence, our
# feed-forward model attempts to predict $x_{t+1}$. During generation, the predicted "token" is fed as the new current
# element.


class FeedForwardModel(nn.Module):
    d_output: int = 256
    d_model: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model)(x)
        x = nn.relu(x)
        x = nn.Dense(self.d_output)(x)
        x = nn.log_softmax(x)
        return x


# We define separate step functions for running training and evaluation steps, accordingly. These step functions are
# each wrapped in a call to `@jax.jit` which fuses operations, generally leading to high performance gains. These @jit
# calls will become increasingly important as we optimize S4.


# Note: Jax by default can't JIT a "nn.Module" (it's immutable anyway)...
# unclear if we even need this (@jit might cache it implicitly)
@partial(jax.jit, static_argnums=(2,))
def ff_train_step(state, batch, model):
    def loss_fn(params):
        logits = model.apply({"params": params}, batch[:, :-1])
        loss = np.mean(jax.vmap(cross_entropy_loss)(logits, batch[:, 1:]))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@partial(jax.jit, static_argnums=(2,))
def ff_eval_step(batch, params, model):
    logits = model.apply({"params": params}, batch[:, :-1])
    loss = np.mean(jax.vmap(cross_entropy_loss)(logits, batch[:, 1:]))
    acc = np.mean(jax.vmap(compute_accuracy)(logits, batch[:, 1:]))
    return loss, acc


# ### LSTM Recurrent Model
# Here, we build a simple LSTM sequence model (w/ optional stacked layers). These are fully recurrent
# models, and are initialized with a 0-hidden state, and rolled out for the full sequence length.


class LSTMRecurrentModel(nn.Module):
    d_output: int = 256
    d_model: int = 64

    def __call__(self, x):
        # TODO @Sidd => Implement (stacked) LSTM Model...
        return x


# ## Sanity Checks
# Here we provide examples for training & evaluation our baseline models on the various datasets.


def example_train(
    model_cls,
    model_train_step_fn,
    model_eval_step_fn,
    create_dataset_fn,
    bsz=128,
    epochs=10,
):
    # Set randomness...
    print("[*] Setting Randomness...")
    key = jax.random.PRNGKey(0)
    key, init_rng, dropout_rng = jax.random.split(key, num=3)

    # Create dataset...
    trainloader, testloader, n_classes, seq_len = create_dataset_fn()

    print("[*] Starting Training =>> Initializing Model + Train State...")
    model = model_cls(d_output=n_classes)
    state = create_train_state(model, init_rng, dropout_rng, bsz=bsz, seq_len=seq_len)

    # Loop over epochs
    for epoch in range(epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        train_loss = train_epoch(state, model, trainloader, model_train_step_fn)

        print(f"[*] Running Epoch {epoch + 1} Validation...")
        test_loss, test_acc = validate(
            state.params, model, testloader, model_eval_step_fn
        )

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(
            f"\tTrain Loss: {train_loss:.5f} -- Test Loss: {test_loss:.5f} -- Test"
            f" Accuracy: {test_acc:.4f}\n"
        )


# Train a feed-forward model on the sin(x) dataset.
#
# ```python
# example_train(FeedForwardModel, ff_train_step, ff_eval_step, create_sin_x_dataset)
# ```

# Train a feed-forward model on the sin(ax + b) dataset.
#
# ```python
# example_train(FeedForwardModel, ff_train_step, ff_eval_step, create_sin_ax_b_dataset)
# ```

# Train a feed-forward model on the MNIST dataset.
#
# ```python
# example_train(FeedForwardModel, ff_train_step, ff_eval_step, create_mnist_dataset)
# ```


# # Part 1: The S4 Model


# General Skeleton for S4 --> takes an S4Layer (naive/without-optimization, or fully loaded)
class S4Model(nn.Module):
    layer: nn.Module
    d_output: int = 256
    d_model: int = 64
    n_layers: int = 4
    dropout: float = 0.2

    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        layers, norms, dropouts = [], [], []
        for _ in range(self.n_layers):
            layers.append(self.layer)
            norms.append(nn.LayerNorm())
            dropouts.append(nn.Dropout(self.dropout, deterministic=False))
        self.layers, self.norms, self.dropouts = layers, norms, dropouts
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x):
        def run(x):
            x = self.encoder(x)
            x = x.T
            for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
                z = x
                z = layer(z)
                z = dropout(z)
                x = z + x
                x = norm(x.T).T
            x = x.T
            x = self.decoder(x)
            x = nn.log_softmax(x)
            return x

        return jax.vmap(run)(x)


# S4 training loop (@Sidd - Reconcile all loops if possible?)
# ```python
# def s4_train_step(state, batch, model):
#     def loss_fn(params):
#         logits = model.apply(
#             {"params": params}, batch[:, :-1], rngs={"dropout": dropout_rng}
#         )
#         loss = cross_entropy_loss(
#             logits=logits[:, : batch.shape[1]], labels=batch[:, 1:]
#         )
#         return loss, logits
#
#     grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#     (_, logits), grads = grad_fn(state.params)
#     state = state.apply_gradients(grads=grads)
#     return state
# ```

# ## State Space Models

# Create a state space model.

# A state space model

# $x'(t) = \mathbf{A} x(t) + \mathbf{B} u(t)$

# $y(t) = \mathbf{C} x(t)$

# To simplify everything we will assume the following.
# $\mathbf{A} \in R^{N \times N}, \mathbf{B} \in R^{N \times 1}, \mathbf{C} \in R^{1 \times N}$

# and $u(t) : R \mapsto R$


# ## Discretization to recurrent


# Discretize SSM

# Bilinear transformation

# important todo.

# https://en.wikipedia.org/wiki/Bilinear_transform


def discretize_SSM(A, B, C, step=1):
    I = np.eye(A.shape[0])
    BL = inv((I - (step / 2.0) * A))
    Abar = BL @ (I + (step / 2.0) * A)
    Bbar = (BL * step) @ B
    return Abar, Bbar, C


# This allows us to run the model as a recurrent network.

# $x_{t+1} = \mathbf{\bar{A}} x_t + \mathbf{\bar{B}} u_t$

# $y_t = \mathbf{\bar{C}} x_t$

# RNN


def iterative_SSM(A, B, C, y):
    def f(X, y):
        X = A @ X + (B * y).ravel()
        return X, C @ X

    return jax.lax.scan(f, np.zeros(B.shape[0]), y)[1]


# ## Convolution

# Because each step of the SSM is linear we can
# also compute $y_t$ without a recurrence.

# CNN

# Make a conv filter version of SSM

# $K = (\bar{C} \bar{B}, \bar{C} \bar{A}^1 \bar{B}, \ldots, \bar{C} \bar{A}^{L-1} \bar{B})$

# We call $K \in R^L$ the convolutional filter representation of the discrete SSM.


def K_conv(A, B, C, L):
    return np.array([(C @ power(A, l) @ B).reshape() for l in range(L)])


# We can then compute $y$ by convolution with this $K$.

# This convolution is big though!


def nonCircularConvolution(x, filt):
    return convolve(x, filt, mode="full", method="fft")[: x.shape[0]]


# ## HiPPO matrices

# For this model to work, initialization is really important.


def make_HiPPO(N):
    p = np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    A = p[:, np.newaxis] @ p[np.newaxis, :]
    return -np.tril(A, k=-1) - np.diag(np.arange(1, N + 1) + 1)


# ## Running it

# Create a model
# N1 = 16
# L1 = 64
# LR = 0.1
# A, B, C = make_HiPPO(N1), Param((N1, 1)), Param((1, N1))
# params = [B, C]

# def model(params, y):
#     ssm = discretize_SSM(A, params[0], params[1])
#     # Turn to convolution
#     K = K_conv(*ssm, L1)
#     # Run as convolution
#     return nonCircularConvolution(y, K)


class NaiveS4Layer(nn.Module):
    H: int = 50
    l_max: int = 16
    dropout: float = 0.2
    N: int = 50

    def setup(self):
        self.A = make_HiPPO(self.N).reshape(1, self.N, self.N)
        self.B = self.param("B", nn.initializers.zeros, (self.H, self.N, 1))
        self.C = self.param("C", nn.initializers.zeros, (self.H, 1, self.N))

    def __call__(self, y):
        def create_ssms(A, B, C):
            ssm = discretize_SSM(A, B, C)
            return K_conv(*ssm, self.l_max)

        K = jax.vmap(lambda B, C: create_ssms(self.A, B, C))(self.B, self.C)

        # Run as convolution
        return jax.vmap(nonCircularConvolution)(y, K)


# def run_train():
#
#     # Run the ssm
#     ssm = discretize_SSM(A, params[0], params[1])
#     v = iterative_SSM(*ssm, y)
#
#     plt.plot(x[1:65], v[1:])
#     plt.plot(x[1:65], y[1:])
#     __st.pyplot()


# # Part 2: Doing it Fast

# ## Generating Functions

# The key idea that S4 is going to exploit is generating functions.

# In particular we are going to convert $K$ from a convolution filter

# $K = (\bar{C} \bar{B}, \bar{C} \bar{A}^1 \bar{B}, \ldots, \bar{C} \bar{A}^{L-1} \bar{B})$

# Into a polynomial where each coefficient represents one element of this sequence.

# discSSM = discretize_SSM(A, B, C)

# $\hat{K}(z) = \bar{C} \bar{B}   + \bar{C} \bar{A}^1 \bar{B} z^1 + \ldots + \bar{C} \bar{A}^{L-1} \bar{B}  z^{L-1}$
def K_gen_simple(*ssm, L):
    K = K_conv(*ssm, L)
    return lambda z: np.sum(K * (z ** np.arange(L)))


# If we apply this function at specific values, we can get back the original convolutional filter.
#
# In particular we apply at the roots of unity,

# $$\Omega_L = \{\exp(2 \pi i  \frac{k}{M}) : k \in [L])\}$$


# And then take an inverse fourier transform to get them back


def convFromGen(gen, L):
    r = np.exp((2j * np.pi / L) * np.arange(L))
    atRoots = jax.vmap(gen)(r)
    order = np.array([i if i == 0 else L - i for i in range(L)])
    out = np.fft.ifft(atRoots, L).reshape(L)
    return out[order]


# What was the point of that? Well working with the generating
# function allows us to do some algebraic manipulations to
# reduce elimanate some of the hard terms.

# In particular the main trick we are going to apply is to turn
# the repeated exponentiation into an inverse.

# $\hat{K}(z) = \bar{C} ( I - A^L) (I - A z)^{-1} \bar{B}$


def K_gen_inverse(A, B, C, L):
    I = np.eye(A.shape[0])
    A_L = power(A, L)
    C2 = C @ (I - A_L)
    return lambda z: (C2 @ inv(I - A * z) @ B).reshape()


# K2 = convFromGen(K_gen_inverse(A, B, C, L=16), 16)

# K2.real

# By working with this generating function we will be able to compute our main term.

# ## Diagonal Plus Low Rank

# This generating function allows us to avoid the matrix power. However it replaces
# it with an inverse which is still not great.

# $(C2  (I - \Gamma  z)^{-1}  B)$

# Let's imagine for a second though that $A$ was diagonal. Then you have a nice form

# $\sum_{i} \frac{C_{1, i} B_{i, 1}} { {1 - \Gamma_{ii} z}}$


# Will make a simple function to compute sums of this form.


def cauchy_dot(v, omega, lambd):
    return (v / (omega - lambd)).sum()


# Diagonal is a pretty strong assumption. But we can relax it by allowing for
# a low-rank component as well with $p, q \in C^{N\times 1}$

# $A = \Gamma + p  q^*$

# The Woodbury identity tells us that the inverse of a diagonal plus low-rank
# is equal to a diagonal plus

# https://en.wikipedia.org/wiki/Woodbury_matrix_identity


# $(\Gamma + p  q^*)^{-1} = \Gamma^{-1} + \Gamma^{-1} p (1 + p^* q)^-1 v^* \Gamma^{-1}$


# The math to get there for real is a bit complex, but here is what the function looks like


def K_gen_DPLR(Gamma, p, q, B, Ct, step=1):
    aterm = (Ct.conj().ravel(), q.conj().ravel())
    bterm = (B.ravel(), p.ravel())

    def gen(o):
        f = (2.0 / step) * ((1.0 - o) / (1.0 + o))
        k00 = cauchy_dot(aterm[0] * bterm[0], f, Gamma)
        k01 = cauchy_dot(aterm[0] * bterm[1], f, Gamma)
        k10 = cauchy_dot(aterm[1] * bterm[0], f, Gamma)
        k11 = cauchy_dot(aterm[1] * bterm[1], f, Gamma)
        return (2.0 / (1.0 + o)) * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    return gen


# ## Turning HiPPO to DPLR

# Define CPU asymmetric eigenvalue decomposition
eig_cpu = jax.jit(eig, backend="cpu")


# Make DPLR HiPPO
def make_DPLR_HiPPO(N):
    p = np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    q = p
    A = p[:, np.newaxis] @ q[np.newaxis, :]
    hippo = -np.tril(A, k=-1) - np.diag(np.arange(1, N + 1) + 1)
    S = hippo + 0.5 * A + 0.5 * np.eye(N)

    # Skew symmetric -- @Sidd Note: eig/eigvals not GPU/lax-backed, so call from cpu instead...
    diag, v = eig_cpu(S)
    diag -= 0.5

    return hippo, diag, 0.5 * p, q, v


# ## The Model
class OptimizedS4Layer(nn.Module):
    H: int = 50
    L: int = 16
    N: int = 50
    step: int = 1

    def setup(self):
        self.A, self.Gamma, self.p, self.q, _ = make_DPLR_HiPPO(self.N)
        self.B = self.param("B", nn.initializers.zeros, (self.H, self.N, 1))
        self.C = self.param("C", nn.initializers.zeros, (self.H, 1, self.N))

        Abar, _, Cbar = discretize_SSM(self.A, self.B, self.C, self.step)
        self.Ct = jax.vmap(
            lambda Cbar: (np.eye(self.N) - power(Abar, self.L)).conj().T @ Cbar.ravel()
        )(Cbar)

    def __call__(self, y):
        def create_ssms(B, Ct):
            K_gen = K_gen_DPLR(self.Gamma, self.p, self.q, B, Ct, self.step)
            return convFromGen(K_gen, self.L)

        K = jax.vmap(create_ssms)(self.B, self.Ct)
        return jax.vmap(nonCircularConvolution)(y, K)


# # Part 3: Putting S4 to the Test

# ## Path-X

# L = 16000
# s = S4(L, 2)
# out2 = s.K_gen()
# out2 = convFromGen(out2, L)
# out2
