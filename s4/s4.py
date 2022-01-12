# <center><h1> The Annotated S4 </h1></center>
#
#
# <center>
# <p><a href="https://arxiv.org/abs/2111.00396">Efficiently Modeling Long Sequences with Structured State Spaces</a></p>
# </center>
#
# <center>
# <p> Albert Gu, Karan Goel, and Christopher Ré.</p>
# </center>
# <img src="images/hero.png" width="100%"/>


# *Blog Post by [Sasha Rush](http://rush-nlp.com/) and [Sidd Karamcheti](https://www.siddkaramcheti.com/)*
#
# The [Structured State Space for Sequence
# Modeling](https://arxiv.org/abs/2111.00396) (S4) architecture is a new approach to very
# long-range sequence modeling tasks for vision,
# language, and audio, showing a capacity to capture dependencies over tens
# of thousands of steps. Especially impressive are the model's impressive results on the challenging
# [Long Range Arena](https://github.com/google-research/long-range-arena) benchmark, showing an ability
# to reason over sequences of up to **16,000+** elements with high accuracy.

# <img src="images/table.png" width="100%"/>

# The paper is also a refreshing departure from Transformers and takes
# a very different approach to an important problem-space.  However,
# several of our colleagues have also noted privately (and on
# [twitter](https://twitter.com/sleepinyourhat/status/1468037897446121483)!)
# the difficulty of gaining intuition for the model.  With this goal,
# this blog post is an implementation of the S4 paper that links code
# with explanation from the paper itself in the style of [the annotated
# transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).
# Hopefully it helps you in following the details of the model.

# ## Table of Contents

# - Part 1: [State Space Models](#part-1-state-space-models)
# - Part 2: [Implementing S4](#part-2-implementing-s4)
# - Part 3: [S4 in Practice](#part-3-s4-in-practice)

# Note that this project uses [JAX](https://github.com/google/jax/)
# with the [Flax](https://github.com/google/flax) NN library.  While
# we personally mainly use Torch, the functional nature of JAX is a good
# fit for some of the complexities of S4. We make heavy use of
# [vmap](https://jax.readthedocs.io/en/latest/jax.html#jax.vmap),
# [scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html),
# their [NN
# cousins](https://flax.readthedocs.io/en/latest/flax.linen.html#module-flax.linen.transforms),
# and most importantly
# [jax.jit](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables)
# to compile fast and efficient S4 layers.

from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal
from jax.numpy.linalg import eig, inv, matrix_power
from jax.scipy.signal import convolve


rng = jax.random.PRNGKey(1)


# ## Part 1: State Space Models

# Okay, let's get started – our goal is the efficient
# modeling of long sequences. To do this, we are going to build a
# new neural network layer based on State Space Models. By the end of
# this section we will be able to build and run a model with this layer.
# To get there though, we need some technical background. Let's work
# our way through the background of the paper.

# > The [state space model](https://en.wikipedia.org/wiki/State-space_representation) is defined by this simple equation.
# > It maps a 1-D input signal $u(t)$ to an $N$-D latent state $x(t)$
# > before projecting to a 1-D output signal $y(t)$.
# $$
#   \begin{aligned}
#     x'(t) &= \boldsymbol{A}x(t) + \boldsymbol{B}u(t) \\
#     y(t) &= \boldsymbol{C}x(t) + \boldsymbol{D}u(t)
#   \end{aligned}
# $$
# > Our goal is
# > to simply use the SSM as a black-box representation in a deep
# > sequence model, where $\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C}, \boldsymbol{D}$ are
# > parameters learned by gradient descent.  For the remainder, we will
# > omit the parameter $\boldsymbol{D}$ for exposition (or equivalently,
# > assume $\boldsymbol{D} = 0$  because the term $\boldsymbol{D}u$ can be
# > viewed as a skip connection and is easy to compute).
# >
# > An SSM maps a input $u(t)$ to a state representation vector $x(t)$ and an output $y(t)$.
# > For simplicity, we assume the input and output are one-dimensional, and the state representation
# > is $N$-dimensional. The first equation defines the change in $x(t)$ over time.

# Our SSMs will be defined by three matrices – $\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C}$ – which
# we will learn. For now, we begin with a function a random SSM, to define sizes,


def random_SSM(rng, N):
    a_r, b_r, c_r = jax.random.split(rng, 3)
    A = jax.random.uniform(a_r, (N, N))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return A, B, C


# ### Discrete-time SSM: The Recurrent Representation
#
# > To be applied on a discrete input sequence $(u_0, u_1, \dots )$
# > instead of continuous function $u(t)$, the SSM must be
# > discretized by a **step size** $\Delta$ that represents the
# > resolution of the input.  Conceptually, the inputs $u_k$ can be
# > viewed as sampling an implicit underlying continuous signal $u(t)$,
# > where $u_k = u(k \Delta)$.
# >
# > To discretize the continuous-time SSM, we use
# > the [bilinear method](https://en.wikipedia.org/wiki/Bilinear_transform), which converts the
# > state matrix $\boldsymbol{A}$ into an approximation $\boldsymbol{\overline{A}}$.  The discrete SSM is:
# $$
# \begin{aligned}
#   \boldsymbol{\overline{A}} &= (\boldsymbol{I} - \Delta/2 \cdot \boldsymbol{A})^{-1}(\boldsymbol{I} + \Delta/2 \cdot \boldsymbol{A}) \\
#   \boldsymbol{\overline{B}} &= (\boldsymbol{I} - \Delta/2 \cdot \boldsymbol{A})^{-1} \Delta \boldsymbol{B} \\
#   \boldsymbol{\overline{C}} &= \boldsymbol{C}\\
# \end{aligned}
# $$


def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


# > This equation is now a *sequence-to-sequence* map $u_k \mapsto y_k$ instead of function-to-function.
# > Moreover the state equation is now a recurrence in $x_k$, allowing the discrete SSM to be computed like an RNN.
# > Concretely, $x_k \in \mathbb{R}^N$ can be viewed as a *hidden state* with transition matrix $\boldsymbol{\overline{A}}$.
# $$
# \begin{aligned}
#   x_{k} &= \boldsymbol{\overline{A}} x_{k-1} + \boldsymbol{\overline{B}} u_k\\
#   y_k &= \boldsymbol{\overline{C}} x_k \\
# \end{aligned}
# $$

# As the paper says, this "step" function does look superficially like that of
# an RNN. We can implement this with a
# [scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html)
# in JAX,


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)[1]


# Putting everything together, we can run the SSM
# by first discretizing, then iterating step by step,


def run_SSM(A, B, C, u):
    L = u.shape[0]
    N = A.shape[0]
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)

    # Run recurrence
    return scan_SSM(Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)))


# ### Tangent: A Mechanics Example

#  To gain some intuition and to test our SSM implementation, we pause
#  from machine learning to implement a [classic example from mechanics](https://en.wikipedia.org/wiki/State-space_representation#Moving_object_example).
#
#  In this example, we consider the forward position $y(t)$ of a mass attached to a wall with a spring.
#  Over time, varying force $u(t)$ is applied to this mass. The system is parameterized by mass ($m$),
#  spring constant ($k$), friction constant ($b$). We can relate these with the following differential equation:

# $$\begin{aligned}
# my''(t) = u(t) - by'(t) - ky(t)
# \end{aligned}
# $$


# Rewriting this in matrix form yields an SSM in the following form:

# $$
# \begin{aligned}
# \boldsymbol{A} &= \begin{bmatrix} 0 & 1 \\ -k/m & -b/m \end{bmatrix}  \\
# \boldsymbol{B} & = \begin{bmatrix} 0  \\ 1/m \end{bmatrix} & \boldsymbol{C} = \begin{bmatrix} 1 & 0  \end{bmatrix}  \\
# \end{aligned}
# $$


def example_mass(k, b, m):
    A = np.array([[0, 1], [-k / m, -b / m]])
    B = np.array([[0], [1.0 / m]])
    C = np.array([[1.0, 0]])
    return A, B, C


#  Looking at the $\boldsymbol{C}$, we should be able to convince ourselves that the
#  first dimension of the hidden state is the position (since that becomes $y(t)$).
#  The second dimension is the velocity, as it is impacted by $u(t)$ through
#  $\boldsymbol{B}$. The transition $\boldsymbol{A}$ relates these terms.
#

# We'll set $u$ to be a continuous function of $t$,


@partial(np.vectorize, signature="()->()")
def example_force(t):
    x = np.sin(10 * t)
    return x * (x > 0.5)


#  Let's run this SSM through our code.


def example_ssm():
    # SSM
    ssm = example_mass(k=40, b=5, m=1)

    # L samples of u(t).
    L = 100
    step = 1.0 / L
    ks = np.arange(L)
    u = example_force(ks * step)

    # Approximation of y(t).
    y = run_SSM(*ssm, u)

    # Plotting ---
    import matplotlib.pyplot as plt
    import seaborn
    from celluloid import Camera

    seaborn.set_context("paper")
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    camera = Camera(fig)
    ax1.set_title("Force $u_k$")
    ax2.set_title("Position $y_k$")
    ax3.set_title("Object")
    ax1.set_xticks([], [])
    ax2.set_xticks([], [])

    # Animate plot over time
    for k in range(0, L, 2):
        ax1.plot(ks[:k], u[:k], color="red")
        ax2.plot(ks[:k], y[:k], color="blue")
        ax3.boxplot(
            [[y[k, 0] - 0.04, y[k, 0], y[k, 0] + 0.04]],
            showcaps=False,
            whis=False,
            vert=False,
            widths=10,
        )
        camera.snap()
    anim = camera.animate()
    anim.save("line.gif", dpi=150, writer="imagemagick")


example_ssm()

# <img src="line.gif" width="100%">

# Neat! And that it was just 1 SSM, with 2 hidden states over 100 steps.
# The final model will have had *100s of stacked SSMs* over *thousands of steps*. But first – we
# need to make these models practical to train.

# ### Training SSMs: The Convolutional Representation

# The punchline of this section is that we can turn the "RNN" above into a "CNN"
# by unrolling. Let's go through the derivation.

# > The recurrent SSM is not practical for training on modern hardware
# > due to its sequential nature.  Instead, there is a well-known connection
# > between linear time-invariant (LTI) SSMs and
# > continuous convolutions.  Correspondingly, the recurrent SSM can actually be
# > written as a [discrete convolution](https://en.wikipedia.org/wiki/Convolution#Discrete_convolution).
# >
# > For simplicity let the initial state be $x_{-1} = 0$. Then unrolling  explicitly yields:
# >
# $$
# \begin{aligned}
#   x_0 &= \boldsymbol{\overline{B}} u_0 &
#   x_1 &= \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{B}} u_1 &
#   x_2 &= \boldsymbol{\overline{A}}^2 \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_1 + \boldsymbol{\overline{B}} u_2 & \dots
#   \\
#   y_0 &= \boldsymbol{\overline{C}} \boldsymbol{\overline{B}} u_0 &
#   y_1 &= \boldsymbol{\overline{C}} \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{C}} \boldsymbol{\overline{B}} u_1 &
#   y_2 &= \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^2 \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_1 + \boldsymbol{\overline{C}} \boldsymbol{\overline{B}} u_2
#   & \dots
# \end{aligned}
# $$
# >
# > This can be vectorized into a convolution with an explicit formula for the convolution kernel.
# >
# $$
# \begin{aligned}
#     y_k &= \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^k \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^{k-1} \boldsymbol{\overline{B}} u_1 + \dots + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_{k-1} + \boldsymbol{\overline{C}}\boldsymbol{\overline{B}} u_k
#     \\
#     y &= \boldsymbol{\overline{K}} \ast u
# \end{aligned}
# $$
# >
# $$
# \begin{aligned}
#   \boldsymbol{\overline{K}} \in \mathbb{R}^L  = (\boldsymbol{\overline{C}}\boldsymbol{\overline{B}}, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}\boldsymbol{\overline{B}}, \dots, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}^{L-1}\boldsymbol{\overline{B}})
# \end{aligned}
# $$
# We call $\boldsymbol{\overline{K}}$ the **SSM convolution kernel** or filter.

# Note that this is a *giant* filter. It is the size of the entire sequence!


def K_conv(Ab, Bb, Cb, L):
    return np.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )


# Warning this implementation is naive and unstable. In practice it is
# actually not going to work for more than very small lengths. 
# However we are going to replace it with S4 in Part 2, so for
# now we just keep it around as a placeholder.


# We can compute the result of applying this filter either with a standard direct convolution or
# with a padded (non-circular) [Fast Fourier Transform (FFT)](https://en.wikipedia.org/wiki/Convolution_theorem).
# As the length gets longer the second method will be more efficient,


def non_circular_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]


# The CNN method and the RNN method yield (roughly) the same result,


def test_cnn_is_rnn(N=4, L=16, step=1.0 / 16):
    ssm = random_SSM(rng, N)
    u = jax.random.uniform(rng, (L,))

    # "RNN"
    rec = run_SSM(*ssm, u)

    # "CNN"
    ssmb = discretize(*ssm, step=step)
    conv = non_circular_convolution(u, K_conv(*ssmb, L))

    # Check
    assert np.isclose(rec.ravel(), conv.ravel(), rtol=1e-2, atol=1e-4).all()


# At this point we have all of the machinery used for SSM training. The next
# steps are about 1) making these models stable to train, and 2) making them fast.

# ### Addressing Long-Range Dependencies with HiPPO

# <img src="images/hippo.png" width="100%"/>
#
# > [Prior work](https://arxiv.org/abs/2008.07669) found that the basic SSM actually performs very poorly in
# > practice.  Intuitively, one explanation is that they  suffer from gradients scaling exponentially in the sequence length (i.e., the
# > vanishing/exploding gradients problem).  To address this problem, previous work developed the HiPPO theory of
# > continuous-time memorization.
# >
# > HiPPO specifies a class of certain matrices $\boldsymbol{A} \in \mathbb{R}^{N \times N}$ that when incorporated,
# > allow the state $x(t)$ to memorize the history of the input $u(t)$.
# > The most important matrix in this class is defined by the HiPPO matrix.
# >
# $$
# \begin{aligned}
#   (\text{\textbf{HiPPO Matrix}})
#   \qquad
#   \boldsymbol{A}_{nk}
#   =
#   \begin{cases}
#     (2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\
#     n+1 & \text{if } n = k \\
#     0 & \text{if } n < k
#   \end{cases}
# \end{aligned}
# $$
# >
# > Previous work found that simply modifying an SSM from a random matrix $\boldsymbol{A}$ to HiPPO
# > improved its performance on the sequential MNIST classification benchmark from $50\%$ to $98\%$.


# This matrix is going to be really important, but it is a bit of
# magic. For our purposes we mainly need to know that: 1) we only need to
# calculate it once and 2) it has a nice, simple structure (which we will exploit in
# part 2). Without going into the ODE math, the main takeaway
# is that this matrix aims to remember the past history in the state a
# timescale invariant manner,


def make_HiPPO(N):
    def v(n, k):
        if n > k:
            return np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
        elif n == k:
            return n + 1
        else:
            return 0

    # Do it slow so we don't mess it up :)
    mat = [[v(n, k) for k in range(1, N + 1)] for n in range(1, N + 1)]
    return np.array(mat)

# The very high-level explanation of this matrix.
# 


# ### An SSM Neural Network.

# We now have everything we need to build an SSM neural network layer.
# As defined above, the discrete SSM defines a map from $\mathbb{R}^L
# \to \mathbb{R}^L$, i.e. a 1-D sequence map. We assume that we
# are going to be learning the parameters $B$ and $C$, as well as a
# step size $\Delta$ and a scalar $D$ parameter. The HiPPO matrix is
# used for the transition $A$. We learn the step size in log space.


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


# For the SMM layer most of the work is to build the filter.
# The actual call to the network is just the (huge) convolution we specified above.
#
# Note for Torch users: `setup` in Flax is called each time the parameters are updated.
# This is similar to the
# [Torch parameterizations](https://pytorch.org/tutorials/intermediate/parametrizations.html).


class SSMLayer(nn.Module):
    A: np.DeviceArray  # HiPPO
    N: int
    l_max: int

    def setup(self):
        # SSM parameters
        self.B = self.param("B", nn.initializers.lecun_normal(), (self.N, 1))
        self.C = self.param("C", nn.initializers.lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        # Step parameter
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        step = np.exp(self.log_step)
        ssm = discretize(self.A, self.B, self.C, step=step)
        self.K = K_conv(*ssm, self.l_max)

    def __call__(self, u):
        return non_circular_convolution(u, self.K) + self.D * u


# Since our SSMs operate on scalars, we make $H$ different, stacked copies ($H$ different SSMs!) with
# different parameters. Here we use the [Flax vmap](
# https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.vmap.html)
# method to easily define these copies,


def cloneLayer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1},
        split_rngs={"params": True},
    )


# We then initialize $A$ with the HiPPO matrix, and pass it into the stack of modules above,


def SSMInit(N):
    return partial(cloneLayer(SSMLayer), A=make_HiPPO(N), N=N)


# This SSM Layer can then be put into a standard NN. For instance, here
# we have a Transformer-style stack of residual blocks, each containing the $H$ stacked SSMs.


class SeqInternal(nn.Module):
    layer: nn.Module
    l_max: int
    dropout: float
    d_model: int
    training: bool = True

    def setup(self):
        self.seq = self.layer(l_max=self.l_max)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x, blank):
        x2 = self.seq(x)
        z = self.drop(self.out(self.drop(nn.gelu(x2))))
        return self.norm(z + x)


class SeqModel(nn.Module):
    layer: nn.Module
    d_output: int
    d_model: int
    l_max: int
    n_layers: int
    dropout: float = 0.2
    training: bool = True
    classification: bool = False

    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SeqInternal(
                layer=self.layer,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                l_max=self.l_max,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, None)
        if self.classification:
            x = np.mean(x, axis=0)
        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


BatchSeqModel = nn.vmap(
    SeqModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None},
    split_rngs={"params": False, "dropout": True},
)


# Overall, this defines a sequence-to-sequence map of shape (batch size, sequence length, hidden dimension),
# exactly the signature exposed by related sequence models such as Transformers, RNNs, and CNNs.

# Full code for training is defined in
# [training.py](https://github.com/srush/s4/blob/main/s4/train.py). While
# we now have our main model, it is not fast enough to actually use. The next
# section is all about making this SSM Layer faster – a lot faster!


# ## Part 2: Implementing S4

# Warning: The section has a lot of math. Roughly it boils down to: we
# can compute the filter from Part 1 with a "HiPPO-like" matrix really
# fast. If you are interested, the details are really neat. If not,
# skip to Part 3 for some cool applications like MNIST completion.

# [Skip Button](#part-3-s4-in-practice)


# > The fundamental bottleneck in computing the discrete-time SSM
# > is that it involves repeated matrix multiplication by
# > $\boldsymbol{\overline{A}}$.  For example, computing
# > naively  involves $L$ successive multiplications
# > by $\boldsymbol{\overline{A}}$, requiring $O(N^2 L)$ operations and
# > $O(NL)$ space.

# Specifically recall this function here:


def K_conv_(Ab, Bb, Cb, L):
    return np.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )


# The contribution of S4 is a method for speeding up this function.
# To do this we are going to focus on the case where the SSM
# has special structure. Specifically, Diagonal Plus Low-Rank (DPLR) in complex
#  space.

# **DPLR:** SSM is  $(\boldsymbol{\Lambda} - \boldsymbol{p}\boldsymbol{q}^*, \boldsymbol{B}, \boldsymbol{C})$ for some diagonal $\boldsymbol{\Lambda}$ and vectors $\boldsymbol{p}, \boldsymbol{q}, \boldsymbol{B}, \boldsymbol{C} \in \mathbb{C}^{N \times 1}$.
#
#
# Under this DPLR assumption, S4 overcomes the speed bottleneck in three steps

#
#  1. Instead of computing $\boldsymbol{\overline{K}}$ directly,
#     we compute its spectrum by evaluating its **[truncated generating function](https://en.wikipedia.org/wiki/Generating_function)** .  This  now involves a matrix *inverse* instead of *power*.
#  2. We show that the diagonal matrix case is equivalent to the computation of a **[Cauchy kernel](https://en.wikipedia.org/wiki/Cauchy_matrix)** $\frac{1}{\omega_j - \zeta_k}$.
#  3. We show the low-rank term can now be corrected by applying the **[Woodbury identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)** which reduces $(\boldsymbol{\Lambda} + \boldsymbol{p}\boldsymbol{q}^*)^{-1}$ in terms of $\boldsymbol{\Lambda}^{-1}$, truly reducing to the diagonal case.


# ### Step 1. SSM Generating Functions

# The main step will be switching from computing the sequence to computing its generating function.
# From the paper's appendix:

# > To address the problem of computing powers of $\boldsymbol{\overline{A}}$, we introduce another technique.
# > Instead of computing the SSM convolution filter $\boldsymbol{\overline{K}}$ directly,
# > we introduce a generating function on its coefficients and compute evaluations of it.
# >
# >The *truncated SSM generating function* at node $z$ with truncation $L$ is
# $$
# \hat{\mathcal{K}}_L(z; \boldsymbol{\overline{A}}, \boldsymbol{\overline{B}}, \boldsymbol{\overline{C}}) \in \mathbb{C} := \sum_{i=0}^{L-1} \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^i \boldsymbol{\overline{B}} z^i
# $$


def K_gen_simple(Ab, Bb, Cb, L):
    K = K_conv(Ab, Bb, Cb, L)

    def gen(z):
        return np.sum(K * (z ** np.arange(L)))

    return gen


# > Intuitively, the generating function essentially converts the SSM convolution filter from the time domain to
# > frequency domain. Importantly, it preserves the same information, and the desired SSM convolution filter
# > can be  recovered from evaluations of its
# > [generating function at the roots of unity](https://math.stackexchange.com/questions/3213142/root-of-unity-filter )
# $\Omega = \{ \exp(2\pi \frac{k}{L} : k \in [L] \}$ stably in $O(L \log L)$ operations by applying an
# [FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform),


def conv_from_gen(gen, L):
    # Evaluate at roots of unity
    Omega_L = np.exp((2j * np.pi / L) * np.arange(L))
    atRoots = jax.vmap(gen)(Omega_L)
    # Inverse FFT
    out = np.fft.ifft(atRoots, L).reshape(L)
    # Numpy returns the values out of order.
    order = np.array([i if i == 0 else L - i for i in range(L)])
    return out[order].real


# But in the generating function we can  replace the power with an inverse!
# $$
# \hat{\mathcal{K}}_L(z) = \sum_{i=0}^{L-1} \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^i \boldsymbol{\overline{B}} z^i = \boldsymbol{\overline{C}} (\boldsymbol{I} - \boldsymbol{\overline{A}}^L z^L) (\boldsymbol{I} - \boldsymbol{\overline{A}} z)^{-1} \boldsymbol{\overline{B}} = \boldsymbol{\tilde{C}}  (\boldsymbol{I} - \boldsymbol{\overline{A}} z)^{-1} \boldsymbol{\overline{B}}
# $$

# And for all $z \in \Omega_L$, we have $z^L = 1$ so that term is removed. We then pull this constant
# term into a new $\boldsymbol{\tilde{C}}$. Critically this function does not call `K_conv`,


def K_gen_inverse(Ab, Bb, Cb, L):
    I = np.eye(Ab.shape[0])
    Ab_L = matrix_power(Ab, L)
    Ct = Cb @ (I - Ab_L)
    return lambda z: (Ct @ inv(I - Ab * z) @ Bb).reshape()


# But it gives the same values as `K_conv`,


def test_gen_inverse(L=16, N=4):
    ssm = random_SSM(rng, N)
    b = K_conv(*ssm, L=L)

    a = conv_from_gen(K_gen_inverse(*ssm, L=L), L)
    assert np.isclose(a, b, rtol=1e-2, atol=1e-4).all()


#  In summary, Step 1 allows us to replace the matrix power with an
#  inverse by utilizing a truncated generating function.
#  However this inverse still needs to be calculated $L$
#  times (for each of the roots of unity).

# ### Step 2: Diagonal Case

# The next step to assume special *structure* on the matrix
# $\boldsymbol{A}$ to avoid the inverse.  To begin, let us first
# convert the equation above to use the original SSM matrices. With
# some algebra you can expand the discretization and show:

# $$
# \begin{aligned}
#   \boldsymbol{\tilde{C}}\left(\boldsymbol{I} - \boldsymbol{\overline{A}} \right)^{-1} \boldsymbol{\overline{B}}
#   =
#   \frac{2\Delta}{1+z} \boldsymbol{\tilde{C}} \left[ {2 \frac{1-z}{1+z}} - \Delta \boldsymbol{A} \right]^{-1} \boldsymbol{B}
# \end{aligned}
# $$


# Now imagine $A=\boldsymbol{\Lambda}$ for a diagonal $\boldsymbol{\Lambda}$. Substituting in the discretization
# formula the authors show that the generating function can be written in the following manner:

# $$ \begin{aligned}
# \boldsymbol{\hat{K}}_{\boldsymbol{\Lambda}}(z) & = c(z) \sum_i \cdot \frac{\tilde{C}_i B_i} {(g(z) - \Lambda_{i})} = c(z) \cdot k_{z, \boldsymbol{\Lambda}}(\boldsymbol{\tilde{C}}, \boldsymbol{B}) \\
#  \end{aligned}$$
# where $c$ is a constant, and $g$ is a function of $z$.


# We have effectively replaced an  inverse with a weighted dot product.
# Let's make a small helper function to compute this weight dot product for use.
# Here [vectorize](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.vectorize.html)
# is a decorator that let's us broadcast this function automatically,


@partial(np.vectorize, signature="(c),(),(c)->()")
def cauchy_dot(v, omega, lambd):
    return (v / (omega - lambd)).sum()


# While not important for our implementation, it is worth noting
# that this is a [Cauchy kernel](https://en.wikipedia.org/wiki/Cauchy_matrix)
# and is the subject of many [fast implementations](https://en.wikipedia.org/wiki/Fast_multipole_method).
# On a GPU though, it is efficient enough just to compute it directly.


# ### Step 3: Diagonal Plus Low-Rank

# The final step is to relax the diagonal assumption. In addition to
# the diagonal term we allow a low-rank component with
# $\boldsymbol{p}, \boldsymbol{q} \in \mathbb{C}^{N\times 1}$ such that:

# $$
# \boldsymbol{A} = \boldsymbol{\Lambda} + \boldsymbol{p}  \boldsymbol{q}^*
# $$

# The [Woodbury identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
# tells us that the inverse of a diagonal plus rank-1 term is equal to the
# inverse of the diagonal plus a rank-1 term. Or in math:

# $$ \begin{aligned}
# (\boldsymbol{\Lambda} + \boldsymbol{p}  \boldsymbol{q}^*)^{-1} &= \boldsymbol{\Lambda}^{-1} - \boldsymbol{\Lambda}^{-1} \boldsymbol{p} (1 + \boldsymbol{q}^* \boldsymbol{p})^{-1} \boldsymbol{q}^* \boldsymbol{\Lambda}^{-1}
#  \end{aligned}
# $$

#  There is a bunch of algebra not shown. But it mostly consists of substituting this component in for A,
#  applying the Woodbury identity and distributing terms. We end up with 4 terms that
#  all look like step 2 above:

# $$ \begin{aligned}
# \boldsymbol{\hat{K}}_{DPLR}(z) & = c(z) [k_{z, \Lambda}(\boldsymbol{\tilde{C}}, \boldsymbol{\boldsymbol{B}}) - k_{z, \Lambda}(\boldsymbol{\tilde{C}}, \boldsymbol{\boldsymbol{p}}) (1 - k_{z, \Lambda}(\boldsymbol{q^*}, \boldsymbol{\boldsymbol{p}}) )^{-1} k_{z, \Lambda}(\boldsymbol{q^*}, \boldsymbol{\boldsymbol{B}}) ]
#  \end{aligned}$$


# The code consists of collecting up the terms and applying 4 weighted dot products,


def K_gen_DPLR(Lambda, p, q, B, Ct, step):
    aterm = (Ct.conj().ravel(), q.conj().ravel())
    bterm = (B.ravel(), p.ravel())

    def gen(o):
        g = (2.0 / step) * ((1.0 - o) / (1.0 + o))
        c = 2.0 / (1.0 + o)

        def k(a):
            return cauchy_dot(a, g, Lambda)

        k00 = k(aterm[0] * bterm[0])
        k01 = k(aterm[0] * bterm[1])
        k10 = k(aterm[1] * bterm[0])
        k11 = k(aterm[1] * bterm[1])
        return c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    return gen


# This is our final version of the $K$ function. Now we can check whether it worked.
# First, let's generate a random Diagonal Plus Low Rank matrix,


def random_DPLR(rng, N):
    l_r, p_r, q_r, b_r, c_r = jax.random.split(rng, 5)
    Lambda = jax.random.uniform(l_r, (N,))
    p = jax.random.uniform(p_r, (N,))
    q = jax.random.uniform(q_r, (N,))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return Lambda, p, q, B, C


# We can check that the DPLR method yields the same filter as computing $\boldsymbol{A}$ directly,


def test_gen_dplr(L=16, N=4):
    I = np.eye(4)

    # Create a DPLR A matrix and discretize
    Lambda, p, q, B, C = random_DPLR(rng, N)
    A = np.diag(Lambda) - p[:, np.newaxis] * q[np.newaxis, :]
    Ab, Bb, Cb = discretize(A, B, C, 1.0 / L)
    a = K_conv(Ab, Bb, Cb, L=L)

    # Compare to the DPLR generating function approach.
    Ct = (I - matrix_power(Ab, L)).conj().T @ Cb.ravel()
    b = conv_from_gen(K_gen_DPLR(Lambda, p, q, B, Ct, step=1.0 / L), L)
    assert np.isclose(a, b, rtol=1e-2, atol=1e-4).all()


# ### Turning HiPPO to DPLR

# This approach applies to DPLR matrices, but remember we would like it to also apply to the HiPPO matrix.
#  While not DPLR in its current form, the HiPPO matrix *does have special structure*. It is
#  [Normal](https://en.wikipedia.org/wiki/Normal_matrix) Plus Low-Rank (NPLR). The paper argues that
# this is just as good as DPLR for the purposes of learning an SSM network.

# > The S4 techniques can apply to any matrix $\boldsymbol{A}$ that can be decomposed as *Normal Plus Low-Rank (NPLR)*.
# $$
# >   \boldsymbol{A} = \boldsymbol{V} \boldsymbol{\Lambda} \boldsymbol{V}^* - \boldsymbol{p} \boldsymbol{q}^\top = \boldsymbol{V} \left( \boldsymbol{\Lambda} - \boldsymbol{V}^* \boldsymbol{p} (\boldsymbol{V}^*\boldsymbol{q})^* \right) \boldsymbol{V}^*
# $$
# > for [unitary](https://en.wikipedia.org/wiki/Unitary_matrix) $\boldsymbol{V} \in \mathbb{C}^{N \times N}$, diagonal $\boldsymbol{\Lambda}$, and low-rank factorization $\boldsymbol{p}, \boldsymbol{q} \in \mathbb{R}^{N \times r}$.  An NPLR SSM is therefore [unitarily](https://en.wikipedia.org/wiki/Unitary_matrix) equivalent to some DPLR matrix.


#  For S4, we need to work with a HiPPO matrix for $\boldsymbol{A}$. This requires extracting
#  $\boldsymbol{\Lambda}$ from this decomposition. The appendix of the paper shows this
#  by getting it into  a [skew-symmetric](https://en.wikipedia.org/wiki/Skew-symmetric_matrix)
#  (normal) + low-rank form. We can use this math to get out the DPLR terms,


def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = -make_HiPPO(N)
    # Add in a rank 1 term. Makes it Normal.
    p = 0.5 * np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, np.newaxis] * q[np.newaxis, :]
    # Diagonalize to S to V \Lambda V^*
    Lambda, V = jax.jit(eig, backend="cpu")(S)
    return nhippo, Lambda, p, q, V


# Final sanity check just to make sure those identities hold,


def test_nplr(N=8):
    A2, Lambda, p, q, V = make_NPLR_HiPPO(N)
    p, q = p[:, np.newaxis], q[:, np.newaxis]
    Lambda = np.diag(Lambda)
    Vc = V.conj().T
    A3 = V @ (Lambda - (Vc @ p) @ (Vc @ q.conj()).conj().T) @ Vc
    A4 = V @ Lambda @ Vc - (p @ q.T)
    assert np.allclose(A2, A3, atol=1e-2, rtol=1e-2)
    assert np.allclose(A2, A4, atol=1e-2, rtol=1e-2)


# ## Part 3: S4 in Practice

# That was a lot of work, but now the actual model is concise. In fact
# we are only using four functions:

# 1. `discretize` → Convert SSM to discrete form.
# 2. `K_gen_DPLR` → Truncated generating function when $\boldsymbol{A}$ is DPLR (S4-part)
# 3. `conv_from_gen` → Convert generating function to filter
# 4. `non_circular_convolution` → Run convolution


#  A full S4 Layer is very similar to the simple SSM layer above. The
#  only difference is in the the computation of $\boldsymbol{K}$.
#  Additionally instead of learning $\boldsymbol{C}$, we learn
#  $\boldsymbol{\tilde{C}}$ so we avoid computing powers of
#  $\boldsymbol{A}$. Note as well that in the original paper $\Lambda, p, q$ are
#  also learned. However, in this post, we leave them fixed for simplicity.


class S4Layer(nn.Module):
    A: np.DeviceArray
    p: np.DeviceArray
    q: np.DeviceArray
    Lambda: np.DeviceArray

    N: int
    l_max: int

    def setup(self):
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.Ct = self.param(
            "Ct", lecun_normal(dtype=jax.numpy.complex64), (1, self.N)
        )
        self.log_step = self.param("log_step", log_step_initializer(), (1,))
        step = np.exp(self.log_step)

        K_gen = K_gen_DPLR(
            self.Lambda, self.p, self.q, self.B, self.Ct, step[0]
        )
        self.K = conv_from_gen(K_gen, self.l_max)

    def __call__(self, u):
        return non_circular_convolution(u, self.K) + self.D * u


S4Layer = cloneLayer(S4Layer)

# We initialize the model by computing a DPLR initializer similar to HiPPO,


def S4LayerInit(N):
    _, Lambda, p, q, V = make_NPLR_HiPPO(N)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    A = np.diag(Lambda) - p[:, np.newaxis] @ q[:, np.newaxis].conj().T
    return partial(S4Layer, N=N, A=A, p=p, q=q, Lambda=Lambda)


# ### MNIST Experiments

# Now that we have the model, we can try it out on some MNIST experiments.
# For these experiments we linearize MNIST and just treat it as a sequence of
# pixels.

# The first experiments we ran were on MNIST classification. While
# not in theory a hard problem, treating MNIST like linear sequence
# classification is a bit strange. However in practice, the model
# with $H=256$ and four layers seems to get up near 99% right away.

#        =>> Epoch 9 Metrics ===
#                Train Loss: 0.06640 -- Test Loss: 0.04091 -- Test Accuracy: 0.9878
#                Best Test Loss: 0.04091 -- Best Test Accuracy: 0.9878 at Epoch 9


# A more visually interesting task is generation of MNIST. Here we
# simply feed in a sequence of pixels into the model and have it
# predict the next one like language modeling. With a little
# tweaking, we are able to get the model to an NLL of 0.52 on this
# task with size 512 and 6 layers (~2m parameters).
# The metric used for this task is *[bits per
# dimension](https://paperswithcode.com/sota/image-generation-on-mnist)* which is
# NLL in base 2 for MNIST. A score of 0.52 is ~0.76 BPD which is near PixelCNN++.

#        =>> Epoch 100 Metrics ===
#                Train Loss: 0.53814 -- Test Loss: 0.52482 -- Test Accuracy: 0.8830
#                Best Test Loss: 0.52117 -- Best Test Accuracy: 0.8842 at Epoch 91


#
# We can sample from the model using the CNN implementation. Ideally we would use the
# RNN form, but that would require a bit more plumbing,


def sample_mnist():
    import matplotlib.pyplot as plt
    from flax.training import checkpoints

    model = S4LayerInit(N=64)
    model = partial(
        BatchSeqModel,
        layer=model,
        d_output=256,
        d_model=256,
        n_layers=6,
        l_max=783,
    )
    rng = jax.random.PRNGKey(0)
    state = checkpoints.restore_checkpoint("models/best_84", None)
    model = model(training=False)
    start = np.zeros((1, 784, 1))

    def loop(i, cur):
        cur, rng = cur
        r, rng = jax.random.split(rng)
        out = model.apply({"params": state["params"]}, cur[:, :-1])
        p = jax.random.categorical(rng, out[0, i])
        cur = jax.ops.index_update(cur, (0, i + 1, 0), p)
        return cur, rng

    out = jax.lax.fori_loop(0, 783, jax.jit(loop), (start, rng))[0]
    plt.imshow(out.reshape(28, 28))
    plt.savefig("sample.png")


sample_mnist()

# <img src="images/sample.png" width="100%">

# We can also do prefix-samples. Given the first 300 pixels, try to complete the image.
# S4 is on the left, true on the right.

# <img src="images/im12.png" width="100%">
# <img src="images/im13.png" width="100%">
# <img src="images/im14.png" width="100%">
# <img src="images/im15.png" width="100%">
# <img src="images/im16.png" width="100%">
# <img src="images/im17.png" width="100%">
# <img src="images/im18.png" width="100%">
# <img src="images/im19.png" width="100%">


# ## Conclusion


# Putting together this post inspired lots of thoughts about future
# work in this area. One obvious conclusion is that long-range
# models have all sorts of future applications from acoustic modeling to
# genomic sequences to trajectories (not to mention our shared area of
# NLP). Another is some surprise that linear models can be so effective
# here, while also opening up a range of efficient techniques.
# Finally from a practical level, the transformations in JAX
# make it really nice to implement complex models like this
# in a very concise way (~200 LoC), with similar efficiency and performance!

# We will end by thanking the authors [Albert Gu](http://web.stanford.edu/~albertgu/) and
# [Karan Goel](https://krandiash.github.io/), who were super helpful in
# putting this together, and pointing you again to their
# [paper](https://arxiv.org/abs/2111.00396) and
# [codebase](https://github.com/HazyResearch/state-spaces).
#
# / Cheers – Sasha & Sidd
