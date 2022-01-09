# # The Annotated S4
#
# <h4>
#   <a href="https://arxiv.org/abs/2111.00396" target="_blank">
#       Efficiently Modeling Long Sequences with Structured State Spaces
#   </a>
# </h4>


# > The recent Structured State Space for Sequence Modeling (S4)
# > architecture has been applied to several difficult sequence modeling
# > tasks, showing a remarkable capacity for reasoning over long-term
# > dependencies. The work is part of a line of serveral projects utilizing
# > state space models to model long-term sequences.

# > There are a lot of reasons to be excited by this work. Most notably
# > are the excellent results on the challenging Long Range Arena benchmark.

#  [image]()

# > For me (srush) personally though, the paper is a refreshing departure from
# > Transformer, and brings a very different style to a problem-space that many of
# > us thought we understood very well. Several of my colleagues have also noted
# > privately (and on twitter!) how difficult the paper was to get intuition for.

# > With this goal, this blog post is an Annotated Implementation of
# > the S4 paper.  The text is mainly taken directly from the arxiv
# > version, with some small modification for clarity or editorial
# > judgment. I will change colors when moving between the text and my
# > comments and tangents throughout.

# > Finally, this notebook is written in Numpy / JAX. Generally we
# > tend to use and adore Torch, but we felt that S4 is really a
# > project that shows off some of JAX's strong points. The functional
# > nature of JAX plays extremely nicely with the mathematical
# > descriptions.

# / authors


# # Table of Contents
#
# - Part 0: The Problem
# - Part I: State Space Models
# - Part II: S4 for Training & Inference
# - Part III: Putting S4 to the Test


# # Part 0: Preliminaries

# Todo..., or maybe hide?

from functools import partial
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn
from celluloid import Camera
from flax import linen as nn
from jax.numpy.linalg import eig, inv, matrix_power
from jax.scipy.signal import convolve


rng = jax.random.PRNGKey(1)


def run_test(fn):
    if __name__ == "__main__":
        fn()


def run_example(fn):
    if __name__ == "__main__":
        fn()


seaborn.set_context("paper")

# # Part 1: Background on State-Space Models

# The [state space model](https://en.wikipedia.org/wiki/State-space_representation) is defined by this simple equation.
# It maps a 1-D input signal $u(t)$ to an $N$-D latent state $x(t)$
# before projecting to a 1-D output signal $y(t)$.

# $$
#   \begin{aligned}
#     x'(t) &= \boldsymbol{A}x(t) + \boldsymbol{B}u(t) \\
#     y(t) &= \boldsymbol{C}x(t) + \boldsymbol{D}u(t)
#   \end{aligned}
# $$


# SSMs are broadly used in many scientific disciplines and related to
# latent state models such as Hidden Markov Models (HMM).  Our goal is
# to simply use the SSM as a black-box representation in a deep
# sequence model, where $\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C}, \boldsymbol{D}$ are
# parameters learned by gradient descent.  For the remainder, we will
# omit the parameter $\boldsymbol{D}$ for exposition (or equivalently,
# assume $\boldsymbol{D} = 0$  because the term $\boldsymbol{D}u$ can be
# viewed as a skip connection and is easy to compute.


# An SSM maps a input $u(t)$ to a state representation vector $x(t)$ and an output $y(t)$.
# For simplicity, we assume the input and output are one-dimensional, and the state representation
# is $N$-dimensional. The first equation defines the change in $x(t)$ over time.

# > Concretely, the parameters of the model are  $\mathbf{A} \in \mathbb{R}^{N \times N}, \mathbf{B} \in \mathbb{R}^{N \times 1}, \mathbf{C} \in \mathbb{R}^{1 \times N}, \mathbf{D}\in \mathbb{R}^{1 \times 1}$.


def randomSSM(rng, N):
    "Generate a random SSM of size N"
    a_r, b_r, c_r = jax.random.split(rng, 3)
    A = jax.random.uniform(a_r, (N, N))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return A, B, C


# ## Discrete-time SSM: The Recurrent Representation

# To be applied on a discrete input sequence $(u_0, u_1, \dots )$
# instead of continuous function $u(t)$, the SSM must be
# discretized by a **step size** $\Delta$ that represents the
# resolution of the input.  Conceptually, the inputs $u_k$ can be
# viewed as sampling an implicit underlying continuous signal $u(t)
# $, where $u_k = u(k \Delta)$.


# To discretize the continuous-time SSM, we use
# the [bilinear method](https://en.wikipedia.org/wiki/Bilinear_transform), which converts the
# state matrix $\boldsymbol{A}$ into an approximation $\boldsymbol{\overline{A}}
# $ .  The discrete SSM is

# $$
# \begin{aligned}
#   \boldsymbol{\overline{A}} &= (\boldsymbol{I} - \Delta/2 \cdot \boldsymbol{A})^{-1}(\boldsymbol{I} + \Delta/2 \cdot \boldsymbol{A}) \\
#   \boldsymbol{\overline{B}} &= (\boldsymbol{I} - \Delta/2 \cdot \boldsymbol{A})^{-1} \Delta \boldsymbol{B} \\
#   \boldsymbol{\overline{C}} &= \boldsymbol{C}\\
# \end{aligned}
# $$


def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = inv((I - (step / 2.0) * A))
    Abar = BL @ (I + (step / 2.0) * A)
    Bbar = (BL * step) @ B
    return Abar, Bbar, C


# This equation is now a *sequence-to-sequence* map $u_k \mapsto y_k$ instead of function-to-function.
# Moreover the state equation is now a recurrence in $x_k$,
# allowing the discrete SSM to be computed like an RNN.
# Concretely, $x_k \in \mathbb{R}^N$ can be viewed as a *hidden state* with transition matrix $\boldsymbol{\overline{A}}$.

# $$
# \begin{aligned}
#   x_{k} &= \boldsymbol{\overline{A}} x_{k-1} + \boldsymbol{\overline{B}} u_k\\
#   y_k &= \boldsymbol{\overline{C}} x_k \\
#    \\
# \end{aligned}
# $$


def stepSSM(Ab, Bb, Cb):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return step


def scanSSM(step_fn, u, x0):
    "Map u to y under with a recurrent cell."
    return jax.lax.scan(step_fn, x0, u)[1]


# > Let us put everything together so far to show how to run an SSM.


def runSSM(A, B, C, u):
    L = u.shape[0]
    N = A.shape[0]
    # Discretize
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)
    # Run recurrence
    return scanSSM(stepSSM(Ab, Bb, Cb), u[:, np.newaxis], np.zeros((N,)))


# ### Tangeant: A Mechanics Example

# To gain some intuition and to test our SSM implementation, we pause
# from the paper to implement a [classic example from mechanics](https://en.wikipedia.org/wiki/State-space_representation#Moving_object_example).


# In this example, we consider the forward position $y(t)$ of a mass attached to a wall with a spring.
# Over time, varying force $u(t)$ is applied to this mass. The system is parameterized by mass ($m$),
# spring constant ($b$), friction constant ($k$). We can relate these with the following differential equation.

# $$\begin{aligned}
# my''(t) = u(t) - by'(t) - ky(t)
# \end{aligned}
# $$

# Rewriting this in matrix form yields an SSM,

# $$
# \begin{aligned}
# \boldsymbol{A} &= \begin{bmatrix} 0 & 1 \\ -k/m & -b/m \end{bmatrix} & \\
# \boldsymbol{B} &= \begin{bmatrix} 0  \\ 1/m \end{bmatrix} & \boldsymbol{C} &= \begin{bmatrix} 0 & 1  \end{bmatrix} \\
# \end{aligned}
# $$


def example_mass(k, b, m):
    A = np.array([[0, 1], [-k / m, -b / m]])
    B = np.array([[0], [1.0 / m]])
    C = np.array([[1.0, 9]])
    return A, B, C


# You should be able to convince yourself that the hidden state $x(t)$ in this model is 2D and
# represents the velocity and position of the mass.
# $\boldsymbol{B}$ adds velocity based on the force.
# $\boldsymbol{C}$ returns the current position.
# The transition $\boldsymbol{A}$ updates the state.


# Let's run this SSM through our code.


def example_ssm():
    # SSM with random forces
    L = 50
    t = np.arange(L)
    u = jax.random.uniform(rng, (L,))
    u = np.where(u > 0.95, 20, 0)
    ssm = example_mass(5, 1, 1)
    y = runSSM(*ssm, u)

    # Setup plots
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    camera = Camera(fig)
    ax1.set_title("Force")
    ax2.set_title("Position")
    ax3.set_title("Object")
    ax1.set_xticks([], [])
    ax2.set_xticks([], [])

    # Animate plot over time.
    for k in range(0, L, 2):
        ax1.plot(t[:k], u[:k], color="red")
        ax2.plot(t[:k], y[:k], color="blue")
        ax3.boxplot(
            [[y[k, 0] - 0.04, y[k, 0], y[k, 0] + 0.04]],
            showcaps=False,
            whis=False,
            vert=False,
            widths=10,
        )
        camera.snap()
    anim = camera.animate()
    anim.save("line.gif", dpi=80, writer="imagemagick")


run_example(example_ssm)

pass
# anim = example_ssm()
# anim.save('line.gif', dpi=80, writer='imagemagick')

# ![]('line.gif')


# ## Training SSMs: The Convolutional Representation


# The recurrent SSM is not practical for training on modern hardware
# due to its sequentiality.  Instead, there is a well-known connection
# between linear time-invariant (LTI) SSMs (as we have seen) and
# continuous convolutions.  Correspondingly, it can actually be
# written as a discrete convolution.

# For simplicity let the initial state be \( x_{-1} = 0 \).
# Then unrolling  explicitly yields

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

# This can be vectorized into a convolution with an explicit formula for the convolution kernel.


# $$
# \begin{aligned}
#     y_k &= \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^k \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^{k-1} \boldsymbol{\overline{B}} u_1 + \dots + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_{k-1} + \boldsymbol{\overline{C}}\boldsymbol{\overline{B}} u_k
#     \\
#     y &= \boldsymbol{\overline{K}} \ast u %
# \end{aligned}

# $$
# \begin{aligned}
#   \boldsymbol{\overline{K}} \in \mathbb{R}^L  = (\boldsymbol{\overline{C}}\boldsymbol{\overline{B}}, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}\boldsymbol{\overline{B}}, \dots, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}^{L-1}\boldsymbol{\overline{B}})
# \end{aligned}
# $$

# We call $\boldsymbol{\overline{K}}$ the **SSM convolution kernel** or filter.


def K_conv(A, B, C, L):
    return np.array([(C @ matrix_power(A, l) @ B).reshape() for l in range(L)])


# In other words, equation is a single (non-circular) convolution and can be computed very efficiently with FFTs, *provided* that $\boldsymbol{\overline{K}}$ is known.


def nonCircularConvolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]


# > We can convince ourselves that the two methods yield the same result by checking explicitly.


def test_cnn_is_rnn(N=4, L=16, step=1.0 / 16):
    ssm = randomSSM(rng, N)
    u = np.arange(L)

    # Recurrent
    rec = runSSM(*ssm, u)

    # Convolution
    ssmb = discretize(*ssm, step=step)
    conv = nonCircularConvolution(u, K_conv(*ssmb, L))
    assert np.isclose(rec.ravel(), conv.ravel(), rtol=1e-2, atol=1e-4).all()


# ## Addressing Long-Range Dependencies with HiPPO

# Prior work found that the basic SSM actually performs very poorly in
# practice.  Intuitively, one explanation is that linear first-order
# ODEs solve to an exponential function, and thus may suffer from
# gradients scaling exponentially in the sequence length (i.e., the
# vanishing/exploding gradients problem).  To address this problem,
# previous work developed the HiPPO theory of continuous-time
# memorization.

# HiPPO specifies a class of certain matrices $\boldsymbol{A} \in \mathbb{R}^{N \times N}$ that when incorporated, allow the state $x(t)$ to memorize the history of the input $u(t)$.
# The most important matrix in this class is defined by the HiPPO matrix.

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


# Previous work found that simply modifying an SSM from a random matrix $\boldsymbol{A}$ to HiPPO improved its performance on the sequential MNIST benchmark from $50\%$ to $98\%$.

# ### Tangent: A First SSM Network.

# We now have everything we need to build an SSM neural network layer.
# As defined above, the discrete SSM defines a map from $\mathbb{R}^L
# \to \mathbb{R}^L$, i.e. a 1-D sequence map. We assume that we
# are going to be learning the parameters $B$ and $C$, as well as a
# step size $\Delta$ and a scalar $D$ paramter.  The HiPPO matrix is
# used for the transition $A$.


class NaiveSSMLayer(nn.Module):
    A: np.DeviceArray
    N: int
    l_max: int
    # Ignored
    d_model: int

    def setup(self):
        self.B = self.param("B", nn.initializers.lecun_normal(), (self.N, 1))
        self.C = self.param("C", nn.initializers.lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.log_step = self.param("log_step", nn.initializers.zeros, (1,))

        # Note for Torch users: `setup` is called each time the
        # parameters are updated. Similar to Torch parameterizations.
        step = np.exp(self.log_step) * 1.0 / self.l_max
        ssm = discretize(self.A, self.B, self.C, step=step)
        self.K = K_conv(*ssm, self.l_max)

    def __call__(self, u):
        return nonCircularConvolution(u, self.K) + self.D * u


# Typically, DNNs operate on feature maps of size $H$ instead of $1$.
# We handle multiple features by simply defining $H$ independent copies of itself.


# Flax method for defining $H$ identical copies.
def cloneLayer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1},
        split_rngs={"params": True},
    )


def NaiveSSMInit(N):
    return partial(cloneLayer(NaiveSSMLayer), A=make_HiPPO(N), N=N)


# Overall, we defines a sequence-to-sequence map of shape (batch size, sequence length, hidden dimension), exactly the same as related sequence models such as Transformers, RNNs, and CNNs.

# > Full code for this is defined in *training.py*


# # Part 2: Doing it Fast with S4


# The fundamental bottleneck in computing the discrete-time SSM
#  is that it involves repeated matrix multiplication by
# $\boldsymbol{\overline{A}}$.  For example, computing
# naively  involves $L$ successive multiplications
# by $\boldsymbol{\overline{A}}$, requiring $O(N^2 L)$ operations and
# $O(NL)$ space.


# > The contribution of S4 is speeding this up by changing the way the
# > operations above are computed. In particular this section is a
# > lot of clever math that applies under a structured parameterization of the
# > model. Specifically:

# The S4 techniques apply to any matrix $\boldsymbol{A}$ that can be decomposed as *Normal Plus Low-Rank (NPLR)*.
# $$
#   \boldsymbol{A} = \boldsymbol{V} \boldsymbol{\Lambda} \boldsymbol{V}^* - \boldsymbol{p} \boldsymbol{q}^\top = \boldsymbol{V} \left( \boldsymbol{\Lambda} - \boldsymbol{V}^* \boldsymbol{p} (\boldsymbol{V}^*\boldsymbol{q})^* \right) \boldsymbol{V}^*
# $$
# for unitary $\boldsymbol{V} \in \mathbb{C}^{N \times N}$, diagonal $\boldsymbol{\Lambda}$, and low-rank factorization $\boldsymbol{p}, \boldsymbol{q} \in \mathbb{R}^{N \times r}$.  An NPLR SSM is therefore unitarily equivalent to some Diagonal Plus Low Rank (DPLR) $(\boldsymbol{\Lambda} - \boldsymbol{p}\boldsymbol{q}^*, \boldsymbol{B}, \boldsymbol{C})$ for some diagonal $\boldsymbol{\Lambda}$ and vectors $\boldsymbol{p}, \boldsymbol{q}, \boldsymbol{B}, \boldsymbol{C} \in \mathbb{C}^{N \times 1}$.


# Under this DPLR assumption, we can overcome this speed bottleneck by
# simultaneously applying three new techniques.
#
#  1.  Instead of computing $\boldsymbol{\overline{K}}$ directly,
#     we compute its spectrum by evaluating its **truncated generating function**  at the roots of unity.
#     $\boldsymbol{\overline{K}}$ can then be found by applying an inverse FFT.  This generating function is closely related to the matrix resolvent, and now involves a matrix *inverse* instead of *power*.
#  2. We show that the diagonal matrix case is equivalent to the computation of a **Cauchy kernel** $\frac{1}{\omega_j - \zeta_k}$.
#  3. We show the low-rank term can now be corrected by applying the **Woodbury identity** which reduces $(\boldsymbol{\Lambda} + \boldsymbol{p}\boldsymbol{q}^*)^{-1}$ in terms of $\boldsymbol{\Lambda}^{-1}$, truly reducing to the diagonal case.
#

# Finally we note the all HiPPO matrices have this NPLR representation. We can therefore find extract a unitarily equivalent DPLR parameterization.


# ## Step 1. SSM Generating Functions

# To address the problem of computing powers of $\boldsymbol{\overline{A}}$, we introduce another technique.
# Instead of computing the SSM convolution filter $\boldsymbol{\overline{K}}$ directly,
# we introduce a [generating function]() on its coefficients and compute evaluations of it.

# The *truncated SSM generating function* at node $z$ with truncation $L$ is


# $$
# \hat{\mathcal{K}}_L(z; \boldsymbol{\overline{A}}, \boldsymbol{\overline{B}}, \boldsymbol{\overline{C}}) \in \mathbb{C} := \sum_{i=0}^{L-1} \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^i \boldsymbol{\overline{B}} z^i
# $$


def K_gen_simple(Ab, Bb, Cb, L):
    K = K_conv(Ab, Bb, Cb, L)

    def gen(z):
        return np.sum(K * (z ** np.arange(L)))

    return gen


# Intuitively, the generating function essentially converts the SSM convolution filter from the time domain to frequency domain.
# Importantly, it preserves the same information, and the desired SSM convolution filter can be recovered from evaluations of its generating function at the roots of unity $\Omega = \{ \exp(2\pi \frac{k}{L} : k \in [L] \}$ stably in $O(L \log L)$ operations by applying a [fast fourier transform]().


def convFromGen(gen, L):
    # Evaluate at roots of unity
    Omega_L = np.exp((2j * np.pi / L) * np.arange(L))
    atRoots = jax.vmap(gen)(Omega_L)
    # Inverse FFT
    out = np.fft.ifft(atRoots, L).reshape(L)
    # Numpy returns the values out of order.
    order = np.array([i if i == 0 else L - i for i in range(L)])
    return out[order].real


# > We can check they return the same thing.


def test_gen(L=16):
    ssm = randomSSM(rng, 4)
    # Convolutional filter
    b = K_conv(*ssm, L=L)

    # From truncated generating function.
    a = convFromGen(K_gen_simple(*ssm, L=L), L)
    assert np.isclose(a, b, rtol=1e-2, atol=1e-4).all()


# Now we can take one more step to switch the power for an inverse.


# $$
# \hat{\mathcal{K}}_L(z) = \sum_{i=0}^{L-1} \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^i \boldsymbol{\overline{B}} z^i = \boldsymbol{\overline{C}} (\boldsymbol{I} - \boldsymbol{\overline{A}}^L z^L) (\boldsymbol{I} - \boldsymbol{\overline{A}} z)^{-1} \boldsymbol{\overline{B}} = \boldsymbol{\tilde{C}}  (\boldsymbol{I} - \boldsymbol{\overline{A}} z)^{-1} \boldsymbol{\overline{B}}
# $$

# For all $z \in \Omega_L$, we have $z^L = 1$ so that term is removed. We then pull this constant term into $\boldsymbol{\tilde{C}}$.

# We can compute the generating function now without building the convolution filter.


def K_gen_inverse(Ab, Bb, Cb, L):
    I = np.eye(Ab.shape[0])
    Ab_L = matrix_power(Ab, L)
    Ct = Cb @ (I - Ab_L)
    return lambda z: (Ct @ inv(I - Ab * z) @ Bb).reshape()


# > Check that it returns the same result.


def test_gen_inverse():
    ssm = randomSSM(rng, 4)
    b = K_conv(*ssm, L=16)

    a = convFromGen(K_gen_inverse(*ssm, L=16), 16)
    assert np.isclose(a, b, rtol=1e-2, atol=1e-4).all()


# ## Step 2: Diagonal Case

# > Step 1 allows us to replace the matrix power with an
# > inverse. However this inverse still needs to be calculated $L$
# > times (for each of the roots of unity).  S4 gets around
# > this issue is to assume special structure on the matrix A.


# > To begin let us first convert the equation above to use the original
# > SSM matrices. With some algebra you can show,

# $$
# \begin{aligned}
#   \boldsymbol{\tilde{C}}\left(\boldsymbol{I} - \boldsymbol{\overline{A}} \right)^{-1} \boldsymbol{\overline{B}}
#   =
#   \frac{2\Delta}{1+z} \boldsymbol{\tilde{C}} \left[ {2 \frac{1-z}{1+z}} - \Delta \boldsymbol{A} \right]^{-1} \boldsymbol{B}
# \end{aligned}
# $$


# > Now imagine $A=\boldsymbol{\Lambda}$ for a diagonal $\boldsymbol{\Lambda}$. Substituting in the discretization formula the authors
# > show that the generating function can be written in the following manner,

# $$ \begin{aligned}
# \boldsymbol{\hat{K}}_{\boldsymbol{\Lambda}}(z) & = c(z) \sum_i \cdot \frac{\tilde{C}_i B_i} {(g(z) - \Lambda_{i})} = c(z) \cdot k_{z, \boldsymbol{\Lambda}}(\boldsymbol{\tilde{C}}, \boldsymbol{B}) \\
#  \end{aligned}$$
# where $c$ is a constant, and $g$ is a function of $z$.


# > We have effectively replaced an  inverse with a weighted dot product.
# > Let's make a small helper function to compute it for us.


@partial(np.vectorize, signature="(c),(),(c)->()")
def cauchy_dot(v, omega, lambd):
    return (v / (omega - lambd)).sum()


# While not important for our implementation, it is worth noting that this is a [Cauchy
# kernel]() and is the subject of many fast implementations. On GPU though, it is
# efficient enough just to compute it directly.


# ## Step 3: Diagonal Plus Low-Rank

# > Next let us relax the diagonal assumption. We  allow for
# > a low-rank component with $\boldsymbol{p}, \boldsymbol{q} \in \mathbb{C}^{N\times 1}$

# $$
# \boldsymbol{A} = \boldsymbol{\Lambda} + \boldsymbol{p}  \boldsymbol{q}^*
# $$

# > The [Woodbury
# > identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
# > tells us that the inverse of a diagonal plus rank-1 term is equal to the
# > inverse of the diagonal plus a rank-1 term.

# $$ \begin{aligned}
# (\boldsymbol{\Lambda} + \boldsymbol{p}  \boldsymbol{q}^*)^{-1} &= \boldsymbol{\Lambda}^{-1} - \boldsymbol{\Lambda}^{-1} \boldsymbol{p} (1 + \boldsymbol{q}^* \boldsymbol{p})^{-1} \boldsymbol{q}^* \boldsymbol{\Lambda}^{-1}
#  \end{aligned}
# $$

# > Substituting in our above terms and distributed, gives 4 weighted dot products.

# $$ \begin{aligned}
# \boldsymbol{\hat{K}}_{DPLR}(z) & = c(z) [k_{z, \Lambda}(\boldsymbol{\tilde{C}}, \boldsymbol{\boldsymbol{B}}) - k_{z, \Lambda}(\boldsymbol{\tilde{C}}, \boldsymbol{\boldsymbol{p}}) (1 - k_{z, \Lambda}(\boldsymbol{q^*}, \boldsymbol{\boldsymbol{p}}) )^{-1} k_{z, \Lambda}(\boldsymbol{q^*}, \boldsymbol{\boldsymbol{B}}) ]
#  \end{aligned}$$


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


# ### Testing: A Structured SSM

# > Now we can check whether this worked. First we generate a random Diaonal Plus Low Rank matrix.


def randomSSSM(rng, N):
    l_r, p_r, q_r, b_r, c_r = jax.random.split(rng, 5)
    Lambda = jax.random.uniform(l_r, (N,))
    p = jax.random.uniform(p_r, (N,))
    q = jax.random.uniform(q_r, (N,))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return Lambda, p, q, B, C


# > New we check that the DPLR method yields the same filter as computing $\boldsymbol{A}$ directly.


def test_gen_dplr():
    L = 16
    I = np.eye(4)

    # Create a DPLR A matrix and discretize
    Lambda, p, q, B, C = randomSSSM(rng, 4)
    A = np.diag(Lambda) - p[:, np.newaxis] * q[np.newaxis, :]
    Ab, Bb, Cb = discretize(A, B, C, 1.0 / L)
    a = K_conv(Ab, Bb, Cb, L=L)

    # Compare to the DPLR generating function approach.
    Ct = (I - matrix_power(Ab, L)).conj().T @ Cb.ravel()
    b = convFromGen(K_gen_DPLR(Lambda, p, q, B, Ct, step=1.0 / L), L)
    assert np.isclose(a, b, rtol=1e-2, atol=1e-4).all()


# ## Turning HiPPO to DPLR

# > Finally recall that we want to work with a HiPPO matrix for
# > $\boldsymbol{A}$. This requires showing that the matrix is NPLR. The
# > easiest way to show that it is normal is to show that it is
# > skew-symmetric which implies that it has complex eigenvalues. The
# > corresponding eigenvectors make up the unitary $\boldsymbol{V}$ matrix.

# $$
#   \boldsymbol{A} = \boldsymbol{V} \boldsymbol{\Lambda} \boldsymbol{V}^* - \boldsymbol{p} \boldsymbol{q}^\top = \boldsymbol{V} \left( \boldsymbol{\Lambda} - \boldsymbol{V}^* \boldsymbol{p} (\boldsymbol{V}^*\boldsymbol{q})^* \right) \boldsymbol{V}^*
# $$


def make_NPLR_HiPPO(N):
    # Make -HiPPO
    hippo = -make_HiPPO(N)

    # Add in a rank 1 term. Makes it normal
    p = 0.5 * np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = hippo + p[:, np.newaxis] * q[np.newaxis, :]

    # Diagonalize to S to V^* \Lambda V
    Lambda, V = jax.jit(eig, backend="cpu")(S)

    return hippo, Lambda, p, q, V


# Let's check just to make sure that the identity holds.


def test_nplr():
    N = 8
    A2, Lambda, p, q, V = make_NPLR_HiPPO(N)
    p, q = p[:, np.newaxis], q[:, np.newaxis]
    Lambda = np.diag(Lambda)
    Vc = V.conj().T
    A3 = V @ (Lambda - (Vc @ p) @ (Vc @ q.conj()).conj().T) @ Vc
    A4 = V @ Lambda @ Vc - (p @ q.T)
    assert np.allclose(A2, A3, atol=1e-2, rtol=1e-2)
    assert np.allclose(A2, A4, atol=1e-2, rtol=1e-2)


# # Part 3: Putting S4 to the Test

# ## The Model

# > A full S4 Layer is roughly similar to the simple SSM layer
# > above. The only difference is in the the computation of $\boldsymbol{K}$
# > which is now done through the structured simplification of the
# > generating function.


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


class S4Layer(nn.Module):
    # Constants
    A: np.DeviceArray
    p: np.DeviceArray
    q: np.DeviceArray
    Lambda: np.DeviceArray

    #
    N: int
    d_model: int
    l_max: int

    def setup(self):
        # self.step = 1.0 / self.l_max
        self.B = self.param("B", nn.initializers.lecun_normal(), (self.N, 1))
        # self.C = self.param("C", nn.initializers.lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))
        # self.Lambda2 = self.param("Lambda2", start(self.Lambda))
        # self.p2 = self.param("p2", start(self.p), (self.N))
        # self.q2 = self.param("q2", start(self.q), (self.N))
        # self.Lambda2 = self.param("Lambda2", nn.initializers.zeros, (self.N), jax.numpy.complex64)
        # self.p2 = self.param("p2", nn.initializers.zeros, (self.N), jax.numpy.complex64)
        # self.q2 = self.param("q2", nn.initializers.zeros, (self.N), jax.numpy.complex64)
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        # Recomputed each time.
        step = np.exp(self.log_step)
        I = np.eye(self.N)
        # Abar, _, Cbar = discretize(self.A, self.B, self.C, step)
        # self.Ct = (I - matrix_power(Abar, self.l_max)).conj().T @ Cbar.ravel()
        self.Ct = self.param(
            "Ct", nn.initializers.lecun_normal(dtype=jax.numpy.complex64), (1, self.N)
        )

        K_gen = K_gen_DPLR(self.Lambda, self.p, self.q, self.B, self.Ct, step[0])
        self.K = convFromGen(K_gen, self.l_max)

    def __call__(self, u):
        return nonCircularConvolution(u, self.K) + self.D * u


# > Repeat layer $H$ times.

S4Layer = cloneLayer(S4Layer)


# > To initialize the model we compute the DPLR unitary equivalent of HiPPO and pass it in.


def S4LayerInit(N):
    # Factor hippo into a unitary transform of a DPLR
    _, Lambda, p, q, V = make_NPLR_HiPPO(N)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    A = np.diag(Lambda) - p[:, np.newaxis] @ q[:, np.newaxis].conj().T
    return partial(S4Layer, N=N, A=A, p=p, q=q, Lambda=Lambda)
