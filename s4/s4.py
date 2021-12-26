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

import jax
import jax.numpy as np
from flax import linen as nn
from jax.numpy.linalg import eig, inv, matrix_power
from jax.scipy.signal import convolve
from functools import partial

# ## Simple Sequence Modeling Datasets
# To show how S4 behaves on various sequence modeling tasks, we create three simple datasets, ranging from a simple toy
# overfitting example, to a more complex $sin$ function tracing problem, finally ending with an MNIST image generation
# task.


# # Part 1: The S4 Model


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

def discretize_SSM(A, B, C, step):
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
    return np.array([(C @ matrix_power(A, l) @ B).reshape() for l in range(L)])


# We can then compute $y$ by convolution with this $K$.

# This convolution is big though!

# @partial(np.vectorize, signature="(c),(c)->(c)")
def nonCircularConvolution(x, filt):
    return convolve(x, filt, mode="full")[: x.shape[0]]


# ## Running it

class NaiveSSMLayer(nn.Module):
    A : np.DeviceArray
    N : int
    d_model : int
    l_max: int 
    dropout: float = 0.2
    

    def setup(self):
        self.B = self.param("B", nn.initializers.lecun_normal(), (self.N, 1))
        self.C = self.param("C", nn.initializers.lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))
        
    def __call__(self, y):
        ssm = discretize_SSM(self.A, self.B, self.C, step=1. / self.l_max)
        K = K_conv(*ssm, self.l_max)
        return nonCircularConvolution(y, K) + self.D * y

NaiveSSMLayer = nn.vmap(
    NaiveSSMLayer,
    in_axes=1,
    out_axes=1,
    variable_axes={"params": 1},
    split_rngs={"params": True},
)

    
# This creates d_output number of SSM layers


# ## HiPPO matrices

# For this model to work, initialization is really important.


def make_HiPPO(N):
    p = np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    A = p[:, np.newaxis] @ p[np.newaxis, :]
    return np.tril(A, k=-1) + np.diag(np.arange(1, N + 1) + 1)



def NaiveSSMInit(N):
    return partial(NaiveSSMLayer, A=make_HiPPO(N), N=N)

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
    return out[order].real


# What was the point of that? Well working with the generating
# function allows us to do some algebraic manipulations to
# reduce elimanate some of the hard terms.

# In particular the main trick we are going to apply is to turn
# the repeated exponentiation into an inverse.

# $\hat{K}(z) = \bar{C} ( I - A^L) (I - A z)^{-1} \bar{B}$


def K_gen_inverse(A, B, C, L):
    I = np.eye(A.shape[0])
    A_L = matrix_power(A, L)
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

@partial(np.vectorize, signature="(c),(),(c)->()")
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

def K_gen_DPLR(Gamma, p, q, B, Ct, step):
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
    A = p[:, np.newaxis] * q[np.newaxis, :]
    hippo = -np.tril(A, k=-1) - np.diag(np.arange(1, N + 1) + 1)
    S = hippo + 0.5 * A + 0.5 * np.eye(N)
    diag, v = eig_cpu(S)
    diag = diag - 0.5
    # Skew symmetric -- @Sidd Note: eig/eigvals not GPU/lax-backed, so call from cpu instead...

    return hippo, diag, 0.5 * p, q, v


# ## The Model
class S4Layer(nn.Module):
    A : np.DeviceArray
    p : np.DeviceArray
    q : np.DeviceArray
    Gamma : np.DeviceArray
    N : int
    d_model: int 
    l_max: int

    def setup(self):
        self.step = 1. / self.l_max
        self.B = self.param("B", nn.initializers.lecun_normal(), (self.N, 1))
        self.C = self.param("C", nn.initializers.lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))
        Abar, _, Cbar = discretize_SSM(self.A, self.B, self.C, self.step)
        I = np.eye(self.N)
        self.Ct = (I - matrix_power(Abar, self.l_max)).conj().T @ Cbar.ravel()

    def __call__(self, y):
        K_gen = K_gen_DPLR(self.Gamma, self.p, self.q, self.B, self.Ct, self.step)
        K = convFromGen(K_gen, self.l_max)
        return nonCircularConvolution(y, K) + self.D * y

S4Layer = nn.vmap(
    S4Layer,
    in_axes=1,
    out_axes=1,
    variable_axes={"params": 1},
    split_rngs={"params": True},
)

    
def S4LayerInit(N):
    # Factor hippo into a unitary transform of a DPLR
    _, Gamma, p, q, V = make_DPLR_HiPPO(N)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    A = np.diag(Gamma) - p[:, np.newaxis] @ q[:, np.newaxis].conj().T
    # A = np.diag(Gamma) - p @ q.conj().T
    return partial(S4Layer, N=N, A=A, p=p, q=q, Gamma=Gamma)

    
# # Part 3: Putting S4 to the Test

# ## Path-X

# L = 16000
# s = S4(L, 2)
# out2 = s.K_gen()
# out2 = convFromGen(out2, L)
# out2
