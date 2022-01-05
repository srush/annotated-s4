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
import matplotlib.pyplot as plt
import seaborn
from celluloid import Camera
from flax import linen as nn
from jax.numpy.linalg import eig, inv, matrix_power


def run_test(fn):
    if __name__ == "__main__":
        fn()


def run_example(fn):
    if __name__ == "__main__":
        fn()


# Define CPU asymmetric eigenvalue decomposition
eig_cpu = jax.jit(eig, backend="cpu")


seaborn.set_context("paper")

# ## Simple Sequence Modeling Datasets
# To show how S4 behaves on various sequence modeling tasks, we create three simple datasets, ranging from a simple toy
# overfitting example, to a more complex $sin$ function tracing problem, finally ending with an MNIST image generation
# task.


# # Part 1: State-Space Models

# S4 is a [state-space model](https://en.wikipedia.org/wiki/State-space_representation) (SSM). Specifically
# it is a linear, time-invariant model,


# $$
# \begin{aligned}
# x'(t) &= \mathbf{A} x(t) + \mathbf{B} u(t) \\
# y(t) &= \mathbf{C} x(t) + \mathbf{D} u(t) \\
# \end{aligned}
# $$

# An SSM maps a input $u(t)$ to a state representation vector $x(t)$ and an output $y(t)$.
# For simplicity, we assume the input and output are one-dimensional, and the state representation
# is $N$-dimensional. The first equation defines the change in $x(t)$ over time.

# Concretely, the parameters of the model are  $\mathbf{A} \in \mathbb{R}^{N \times N}, \mathbf{B} \in \mathbb{R}^{N \times 1}, \mathbf{C} \in \mathbb{R}^{1 \times N}, \mathbf{D}\in \mathbb{R}^{1 \times 1}$. Following the S4 paper we will assume $\mathbf{D}=0$ and ignore.


def randomSSM(rng, N):
    a_r, b_r, c_r = jax.random.split(rng, 3)
    A = jax.random.uniform(a_r, (N, N))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return A, B, C


# Instead of working with the continuous functions, we approximate the SSM
# into a discrete time-series representation. This acts on specific samples
# of the input sequence.


# This is done by applying a with a [bilinear transformation]
# (https://en.wikipedia.org/wiki/Bilinear_transform). $\Delta$ is a
# step parameter, roughly corresponding to a sampling rate.
#
# $$
# \begin{aligned}
# \mathbf{\bar{A}} &= (I + \frac{\Delta}{ 2} \mathbf{A} ) (I- \frac{\Delta}{ 2} \mathbf{A})^{-1}  \\
# \mathbf{\bar{B}} &= (I- \frac{\Delta}{ 2} \mathbf{A})^{-1} \mathbf{B} \\
# \mathbf{\bar{C}} &= \mathbf{C}
# \end{aligned}
# $$


def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = inv((I - (step / 2.0) * A))
    Abar = BL @ (I + (step / 2.0) * A)
    Bbar = (BL * step) @ B
    return Abar, Bbar, C


# This allows us to run the model as a discrete recurrence.

# $$
# \begin{aligned}
# x_{t+1} &= \mathbf{\bar{A}} x_t + \mathbf{\bar{B}} u_t \\
# y_t &= \mathbf{\bar{C}} x_t + \mathbf{\bar{D}} u_t
# \end{aligned}
# $$

# The code for this looks similar to a recurrent neural network,
# but note that everything here is linear.


def stepSSM(A, B, C):
    def step(x_t, u_t):
        x_t = A @ x_t + B @ u_t
        y_t = C @ x_t
        return x_t, y_t

    return step


def scanSSM(step_fn, u, init):
    assert u.shape[1] == 1
    return jax.lax.scan(step_fn, init, u)[1]


def runSSM(ssm, u):
    step = 1.0 / u.shape[0]
    ssm = discretize(*ssm, step=step)
    return scanSSM(stepSSM(*ssm), u[:, np.newaxis], np.zeros((ssm[0].shape[0],)))


# ### Example: Mass on a spring.

# To test our SSM implementation, let us implement a class SSM
# for mechanics described [here](https://en.wikipedia.org/wiki/State-space_representation#Moving_object_example).

# This example notes that the position $y(t)$ of a mass attached to the wall with a spring and forward force $u(t)$ is represented as,

# $$\begin{aligned}
# my''(t) = u(t) - by'(t) - ky(t)
# \end{aligned}
# $$

# Can be computed with the following SSM. In general we can turn any Nth-order differential equation with a [single
# first order *matrix* differential equation.](https://lpsa.swarthmore.edu/Representations/SysRepSS.html).


def example_mass(k, b, m):
    A = np.array([[0, 1], [-k / m, -b / m]])
    B = np.array([0, 1.0 / m]).reshape(2, 1)
    C = np.array([0, 1]).reshape(1, 2)
    return A, B, C


# We can discretize this and see the output.


def example_ssm():
    L = 50
    t = np.arange(L)
    rng = jax.random.PRNGKey(1)
    u = jax.random.uniform(rng, (L,))
    u = np.where(u > 0.95, 20, 0)
    ssm = example_mass(5, 1, 1)
    y = runSSM(ssm, u)

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    camera = Camera(fig)
    ax1.set_title("Force")
    ax1.set_xticks([], [])
    ax2.set_xticks([], [])
    ax2.set_title("Position")
    ax3.set_title("Object")

    for i in range(0, L, 2):
        ax1.plot(t[:i], u[:i], color="red")
        ax2.plot(t[:i], y[:i], color="blue")
        ax3.boxplot(
            [[y[i, 0] - 0.04, y[i, 0], y[i, 0] + 0.04]],
            showcaps=False,
            whis=False,
            vert=False,
            widths=10,
        )
        camera.snap()
    anim = camera.animate()
    anim.save("line.gif", dpi=80, writer="imagemagick")


run_example(example_ssm)

# ![]('./line.gif')


# ## Convolution

# We can expand the above SSM recurrence in the following manner. Using the models linearity to eliminate direct calculation of $x_l$.

# $$
# \begin{aligned}
# x_{0} &= \mathbf{\bar{B}} u_0\\
# y_{0} &= \mathbf{\bar{C}} \mathbf{\bar{B}} u_0 \\
# x_{1} &= \mathbf{\bar{A}} \mathbf{\bar{B}} u_0  + \mathbf{\bar{B}} u_1\\
# y_{1} &= \mathbf{\bar{C}} \mathbf{\bar{A}} \mathbf{\bar{B}} u_0  + \mathbf{\bar{C}} \mathbf{\bar{B}} u_1 \\
# x_{2} &= \mathbf{\bar{A}}^2 \mathbf{\bar{B}} u_0  + \mathbf{\bar{A}} \mathbf{\bar{B}} u_1 + \mathbf{\bar{B}} u_2\\
# y_{2} &= \mathbf{\bar{C}} \mathbf{\bar{A}}^2 \mathbf{\bar{B}} u_0  + \mathbf{\bar{C}} \mathbf{\bar{A}} \mathbf{\bar{B}} u_1 + \mathbf{\bar{C}} \mathbf{\bar{B}} u_2\\
#  \end{aligned}
# $$

# This implies that $y_l$ will be a function of the $l$ previous $u_{<=l}$ terms and require $\mathbf{\bar{A}}^l$.

# $$y_{n} = \mathbf{\bar{C}} \mathbf{\bar{A}}^l \mathbf{\bar{B}} u_0  + \ldots + \mathbf{\bar{C}} \mathbf{\bar{B}} u_{l}$$

#  In fact, the calculation for $y_l$ and $y_{l-1}$ is
# near identical, with one new term added.

# $$y_{n-1} = \mathbf{\bar{C}} \mathbf{\bar{A}}^{l-1} \mathbf{\bar{B}} u_0  + \ldots + \mathbf{\bar{C}} \mathbf{\bar{B}} u_{l-1}$$

# This calculation can therefore be done with a convolution instead of a recurrence, with the caveat that the convolutional filter
# grows with the sequence length. Assuming the sequence is of length $L$ we compute the convolutional filter $\mathbf{K}\in\mathbb{R}^L$.

# $$
# \begin{aligned}
# \mathbf{K} &= (\mathbf{\bar{C}} \mathbf{\bar{B}}, \mathbf{\bar{C}} \mathbf{\bar{A}}^1 \mathbf{\bar{B}}, \ldots, \mathbf{\bar{C}} \mathbf{\bar{A}}^{L-1} \mathbf{\bar{B}}) \\
# \end{aligned}
# $$


def K_conv(A, B, C, L):
    return np.array([(C @ matrix_power(A, l) @ B).reshape() for l in range(L)])


# We can then compute $y$ by "full" convolution with this $K$. Note: full convolution will include all possible
# overlaps. The first term $y_0 = \mathbf{K}_0 u_0$ and subsequently $y_l = \sum_{t=0}^l \mathbf{K}_t u_{l-t}$.
#
# We throw out terms where $l >= L$. Note that if $L$ is long this convolution should be computed using an Fast Fourier
# Transform (FFT). Depending on your application you can write the simplified convolution and let JAX figure out how to
# compute the convolution for us. However, we note degraded performance as $L$ increases in size, so we default to the
# FFT expression of the convolution.


# ```python
# # Letting JAX figure this out...
# # from jax.scipy.signal import convolve
# # def nonCircularConvolution(u, K):
# #     return convolve(u, K, mode="full")[: u.shape[0]]
# ```

# Convolution using a Fast Fourier Transform
def nonCircularConvolution(u, K):
    ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
    Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
    out = ud * Kd
    return np.fft.irfft(out)[: u.shape[0]]


# We can see that both approaches – the recurrence and the convolution - compute the same value.

# Check that convolution and recurrence return the same thing.
def test_cnn_rnn():
    ssm = randomSSM(jax.random.PRNGKey(0), 4)
    u = np.arange(16)
    L = u.shape[0]
    step = 1.0 / L
    ssm = discretize(*ssm, step=step)

    # Recurrent
    rec = scanSSM(stepSSM(*ssm), u[:, np.newaxis], np.zeros((ssm[0].shape[0],)))

    # Convolution
    conv = nonCircularConvolution(u, K_conv(*ssm, L))
    assert np.isclose(rec.ravel(), conv.ravel(), rtol=1e-2, atol=1e-4).all()


test_cnn_rnn()


# ## HiPPO


# For this model to work, initialization is really important. **TODO :: HiPPO Magic **

# ...


def make_HiPPO(N):
    return np.array(
        [
            [
                np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
                if n > k
                else (n + 1 if n == k else 0.0)
                for k in range(1, N + 1)
            ]
            for n in range(1, N + 1)
        ]
    )


# ## A First SSM Network.


# We now have everything we need to build an SSM neural network layer.
# This layer will let us learn the parameters of B and C along with
# a HiPPO matrix for our transition A.

# We assume that we are going to be learning the parameters $B$ and $C$.
# The main code simply discretizes these and then computes $y$ with a convolution.


class NaiveSSMLayer(nn.Module):
    A: np.DeviceArray
    N: int
    # Max Length L
    l_max: int
    # Ignored
    d_model: int

    def setup(self):
        self.B = self.param("B", nn.initializers.lecun_normal(), (self.N, 1))
        self.C = self.param("C", nn.initializers.lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))
        ssm = discretize(self.A, self.B, self.C, step=1.0 / self.l_max)
        self.K = K_conv(*ssm, self.l_max)

    def __call__(self, u):
        return nonCircularConvolution(u, self.K) + self.D * u


# In practice though we don't just want to run one SSM, but ideally we
# would learn dozens of SSMs. We do this by copying the network structure
# $H$ different times.

NaiveSSMLayer = nn.vmap(
    NaiveSSMLayer,
    in_axes=1,
    out_axes=1,
    variable_axes={"params": 1},
    split_rngs={"params": True},
)


def NaiveSSMInit(N):
    return partial(NaiveSSMLayer, A=make_HiPPO(N), N=N)


# # Part 2: Doing it Fast - S4

# Everything above describes the full model and approach used in
# S4. However it has the problem that computing $\mathbf{K}$ for
# hundreds of layers $H$ with very large $L$ is inefficient. In this
# section we focus exclusively on efficiently computing $\mathbf{K}$.


# ## Step 1: Generating Functions

# The key idea that S4 is going to exploit is truncated [generating functions](https://en.wikipedia.org/wiki/Generating_function).

# In particular we are going to view the convolutional filter $\mathbf{K}$,

# $$
# \begin{aligned}
# \mathbf{K} &= (\mathbf{\bar{C}} \mathbf{\bar{B}}, \mathbf{\bar{C}} \mathbf{\bar{A}}^1 \mathbf{\bar{B}}, \ldots, \mathbf{\bar{C}} \mathbf{\bar{A}}^{L-1} \mathbf{\bar{B}}) \\
# \end{aligned}
# $$


# instead as a polynomial $\mathbf{\hat{K}}$ where each coefficient represents one element of this sequence.

# $$ \begin{aligned}
# \mathbf{\hat{K}}(z) = \mathbf{\bar{C}} \mathbf{\bar{B}} + \mathbf{\bar{C}} \mathbf{\bar{A}}^1 \mathbf{\bar{B}} z^1 + \ldots + \mathbf{\bar{C}} \mathbf{\bar{A}}^{L-1} \mathbf{\bar{B}} z^{l-1} \\
# \end{aligned}
# $$


def K_gen_simple(*ssm, L):
    K = K_conv(*ssm, L)
    return lambda z: np.sum(K * (z ** np.arange(L)))


# If we apply this function at specific values, we can get back the
# original convolutional filter.  In particular, the formula is to apply
# this function at the roots of unity

# $$\Omega_L = \{\exp(2 \pi i  \frac{k}{L}) : k \in [L])\}$$


# And then take inverse fourier transform to recover the original
# coefficients.


def convFromGen(gen, L):
    # Roots of Unity
    r = np.exp((2j * np.pi / L) * np.arange(L))
    atRoots = jax.vmap(gen)(r)

    # Inverse FFT
    out = np.fft.ifft(atRoots, L).reshape(L)

    # Get the order right.
    order = np.array([i if i == 0 else L - i for i in range(L)])
    return out[order].real


# Check that our original (slow) convolution and our generating-function induced convolution return the same values.


def test_gen():
    ssm = randomSSM(jax.random.PRNGKey(0), 4)
    b = K_conv(*ssm, L=16)

    a = convFromGen(K_gen_simple(*ssm, L=16), 16)
    assert np.isclose(a, b, rtol=1e-2, atol=1e-4).all()


test_gen()


# What was the point of that? Well working with the generating
# function allows us to do some algebraic manipulations to
# eliminate some of the hard terms.

# In particular the main trick is to turn the repeated exponentiation into an inverse. The algebraic manipulations
# in the paper lead to the following extremely reduced functional form.

# $$ \begin{aligned}
# \mathbf{\hat{K}}(z) = \mathbf{\bar{C}} ( I - \mathbf{A}^L z^L) (I - \mathbf{A} z)^{-1} \mathbf{\bar{B}}
#  \end{aligned}$$

# Furthermore when applied at the roots of unity for $L$ the $z^L$ term goes away, as it reduces to 1.


def K_gen_inverse(A, B, C, L):
    I = np.eye(A.shape[0])
    A_L = matrix_power(A, L)
    C2 = C @ (I - A_L)
    return lambda z: (C2 @ inv(I - A * z) @ B).reshape()


# We can verify that this reduction is equivalent to the original convolution.


def test_gen_inverse():
    ssm = randomSSM(jax.random.PRNGKey(0), 4)
    b = K_conv(*ssm, L=16)

    a = convFromGen(K_gen_inverse(*ssm, L=16), 16)
    assert np.isclose(a, b, rtol=1e-2, atol=1e-4).all()


test_gen_inverse()

# ## Step 2: Diagonal Plus Low Rank

# Moving to generating function allows us to replace the matrix power
# with an inverse. However this inverse still needs to be calculated
# $L$ times (for each of the roots of unity).

# $$ \begin{aligned}
# \mathbf{\hat{K}}(z) = \mathbf{\bar{C}} ( I - \mathbf{A}^L) (I - \mathbf{A} z)^{-1} \mathbf{\bar{B}} = \mathbf{\tilde{C}} (I - \mathbf{\bar{A}} z)^{-1} \mathbf{\bar{B}}
#  \end{aligned}$$

# The way S4 gets around this issue is to assume special structure on the
# matrix A.

# First, imagine $A=\Lambda$ for a diagonal $\Lambda$. Substituting in the discretization formula the authors
# show that the generating function can be written in the following manner,

# $$ \begin{aligned}
# \mathbf{\hat{K}}(z) & = \mathbf{\tilde{C}} (I - \mathbf{\bar{A}} z)^{-1} \mathbf{\bar{B}}=   c \cdot \mathbf{\tilde{C}} (g(z) - \mathbf{A})^{-1} \mathbf{\bar{B}} \\
#  &= c \sum_i \cdot \frac{\tilde{C}_i \bar{B}_i} {(g(z) - \Lambda_{i})} = c \cdot k_{z, \Lambda}(\mathbf{\tilde{C}}, \mathbf{\bar{B}}) \\
#  \end{aligned}$$


# This term does not require an inverse and can be computed as a weighted dot product.
# While not important for our implementation, it is worth noting that this is a [Cauchy
# kernel]() and is the subject of many fast implementations. On GPU though, it is
# efficient enough just to compute it directly.


@partial(np.vectorize, signature="(c),(),(c)->()")
def cauchy_dot(v, omega, lambd):
    return (v / (omega - lambd)).sum()


# Next let us relax the diagonal assumption. We  allow for
# a low-rank component with $p, q \in C^{N\times 1}$

# $$A = \Lambda + p  q^*$$

# The [Woodbury identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)
# says that the inverse of a diagonal plus low-rank (DPLR) matrix is equal to the inverse of the diagonal plus a
# low-rank term.

# $$ \begin{aligned}
# (\Lambda + p  q^*)^{-1} &= \Lambda^{-1} - \Lambda^{-1} p (1 + q^* p)^{-1} q^* \Lambda^{-1}
#  \end{aligned}
# $$

# When substituted in these components into the formula above and distributed, the low-rank
# turns it into 4 weighted dot products.

# $$ \begin{aligned}
# \mathbf{\hat{K}}(z) & = c [k_{z, \Lambda}(\mathbf{\tilde{C}}, \mathbf{\mathbf{B}}) - k_{z, \Lambda}(\mathbf{\tilde{C}}, \mathbf{\mathbf{p}}) (1 - k_{z, \Lambda}(\mathbf{q^*}, \mathbf{\mathbf{p}}) )^{-1} k_{z, \Lambda}(\mathbf{q^*}, \mathbf{\mathbf{B}}) ]
#  \end{aligned}$$


# The math to get there for real is a bit complex, but here is what the function looks like


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


# Now we can check whether this worked. First we generate a random Dynamic plus Low-Rank (DPLR) matrix A.


def randomSSSM(rng, N):
    l_r, p_r, q_r, b_r, c_r = jax.random.split(rng, 5)
    Lambda = jax.random.uniform(l_r, (N,))
    p = jax.random.uniform(p_r, (N,))
    q = jax.random.uniform(q_r, (N,))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))
    return Lambda, p, q, B, C


# Now we check that the DPLR method yields the same filter.


def test_gen_dplr():
    L = 16
    I = np.eye(4)

    # Create a DPLR A matrix and discretize
    Lambda, p, q, B, C = randomSSSM(jax.random.PRNGKey(0), 4)
    A = np.diag(Lambda) - p[:, np.newaxis] * q[np.newaxis, :]
    Ab, Bb, Cb = discretize(A, B, C, 1.0 / L)
    a = K_conv(Ab, Bb, Cb, L=L)

    # Compare to the S4 approach.
    Ct = (I - matrix_power(Ab, L)).conj().T @ Cb.ravel()
    b = convFromGen(K_gen_DPLR(Lambda, p, q, B, Ct, step=1.0 / L), L)
    assert np.isclose(a, b, rtol=1e-2, atol=1e-4).all()


test_gen_dplr()


# ## Step 3: Turning HiPPO to DPLR

# This method allows us to compute the generating function for a SSM with A in DPLR form.
# However we are not interested in any A. We want the A matrix to follow the HiPPO formulation.

# It turns out that the HiPPO matrix above is not DPLR. However the paper shows that it is *normal plus low-rank* (NPLR).
# They also show that NPLR matrices are [*unitarily equivalent*](https://en.wikipedia.org/wiki/Specht%27s_theorem) to
# a DPLR matrix.
#
# For our purposes this means that there is some DPLR matrix that is just as good as HiPPO. The paper shows how to
# compute it.


# This function computes all the terms for HiPPO $A$ in equation (6).
#
# $$\mathbf{A} = \mathbf{V} ( \Lambda - (\mathbf{V}^* \mathbf{p}) ( \mathbf{V}^* \mathbf{q})^*) \mathbf{V}^*$$

# Make DPLR HiPPO
def make_DPLR_HiPPO(N):
    # Make HiPPO
    p = np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    q = p
    pq = p[:, np.newaxis] * q[np.newaxis, :]
    hippo = -np.tril(pq, k=-1) - np.diag(np.arange(1, N + 1) + 1)

    # Add in a rank 1 term. Makes it skew-symmetric
    S = hippo + (0.5 * pq + 0.5 * np.eye(N))

    # Diagonalize to S to V^* \Lambda V
    diag, v = eig_cpu(S)
    diag -= 0.5
    return hippo, diag, 0.5 * p, q, v


# Let's check just to make sure that the identity holds.


def test_decomposition():
    N = 8
    A2, Lambda, p, q, V = make_DPLR_HiPPO(N)
    p, q = p[:, np.newaxis], q[:, np.newaxis]
    Lambda = np.diag(Lambda)
    Vc = V.conj().T
    A3 = V @ (Lambda - (Vc @ p) @ (Vc @ q.conj()).conj().T) @ Vc
    A4 = V @ Lambda @ Vc - (p @ q.T)
    assert np.allclose(A2, A3, atol=1e-2, rtol=1e-2)
    assert np.allclose(A2, A4, atol=1e-2, rtol=1e-2)


test_decomposition()


# # Part 3: Putting S4 to the Test


# ## The Model

# Our full S4 Layer is roughly similar to the simple SSM layer above. The only difference
# is in the the computation of $K$ which is now done through the structured simplification
# of the generating function.


class S4Layer(nn.Module):
    A: np.DeviceArray
    p: np.DeviceArray
    q: np.DeviceArray
    Lambda: np.DeviceArray
    N: int
    d_model: int
    l_max: int

    def setup(self):
        self.step = 1.0 / self.l_max
        self.B = self.param("B", nn.initializers.lecun_normal(), (self.N, 1))
        self.C = self.param("C", nn.initializers.lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        # Recomputed each time.
        I = np.eye(self.N)
        Abar, _, Cbar = discretize(self.A, self.B, self.C, self.step)
        self.Ct = (I - matrix_power(Abar, self.l_max)).conj().T @ Cbar.ravel()
        K_gen = K_gen_DPLR(self.Lambda, self.p, self.q, self.B, self.Ct, self.step)
        self.K = convFromGen(K_gen, self.l_max)

    def __call__(self, u):
        return nonCircularConvolution(u, self.K) + self.D * u


S4Layer = nn.vmap(
    S4Layer,
    in_axes=1,
    out_axes=1,
    variable_axes={"params": 1},
    split_rngs={"params": True},
)


# To initialize the model we compute the DPLR unitary equivalent of HiPPO and pass it in.

# $$\mathbf{A} = \Lambda - (\mathbf{V}^* \mathbf{p}) ( \mathbf{V}^* \mathbf{q}) ^*$$


def S4LayerInit(N):
    # Factor HiPPO into a unitary transform of a DPLR
    _, Lambda, p, q, V = make_DPLR_HiPPO(N)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    A = np.diag(Lambda) - p[:, np.newaxis] @ q[:, np.newaxis].conj().T
    return partial(S4Layer, N=N, A=A, p=p, q=q, Lambda=Lambda)


# ## MNIST, CIFAR, & Path-X
