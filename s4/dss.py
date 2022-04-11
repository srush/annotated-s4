# <center><h1> The Diagonal State Space Model </h1></center>
#
#
# <center>
# <p><a href="https://arxiv.org/abs/2203.14343">Diagonal State Spaces are as Effective as Structured State Spaces</a></p>
# </center>
#
# <center>
# <p> Ankit Gupta</p>
#
# ---
#
# *Note: This page is meant as a standalone complement to Section 2 [TODO Link] of the original
# blog post.*
#
# The months following the release of S4 paper by Gu et. al. were characterized by a wave of excitement around the new
# model, it's ability to handle extremely long sequences, and generally, what such a departure from Transformer-based
# architectures could mean. The original authors came out with a
# [follow-up paper applying S4 to audio generation](https://arxiv.org/abs/2202.09729), and weeks later, a completely
# [different group applied S4 to long-range movie clip classification](https://arxiv.org/abs/2204.01692).
#
# Yet, it remains hard to parse aspects of the implementation, especially the derivation of the diagonal plus low rank
# constraint on $\boldsymbol{A}$. Not only was this math fairly complex, but in code, required the use of custom CUDA
# kernels -- further obfuscating the implementation (and why this blog uses Jax to efficiently compile the relevant
# operations).
#
# However, at the end of March 2022 -- an alternative construction for state space models was proposed in [Diagonal
# State Spaces are as Effective as Structured State Spaces](https://arxiv.org/abs/2203.14343). This short paper derives
# an alternative construction of learnable state space models that is both 1) simple, 2) requires no custom kernels, and
# 3) can be efficiently implemented in Jax or PyTorch in just a dozen lines. The rest of this post steps through this
# alternative derivation, **a complete standalone for Section 2** of the original Annotated S4 post.
#
# We'll still be using Jax with the Flax NN Library for consistency with the original post, though this Diagonal State
# Space (DSS) variant can be easily implemented in PyTorch with some minor changes.

# import s4.s4 as s4  TODO -- For some reason breaks streamlit...
import s4
from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal

rng = jax.random.PRNGKey(1)

# ## Table of Contents
# <nav id="TOC">
# <ul>
#   <li>Step 1. The Problem with the SSM Convolutional Kernel
#       <ul>
#           <li>Rethinking Discretization</li>
#           <li>Rewriting the SSM Kernel</li>
#           <li>Diagonalization & Efficient Matrix Powers</li>
#       <ul>
#   </li>
#   <li>Step 2. Deriving the Diagonal State Space Model
#       <ul>
#           <li>Proving Proposition 1 from the DSS Paper</li>
#           <li>Secret Sauce 1: Handling the Complex Softmax</li>
#           <li>Secret Sauce 2: Initializing with the HiPPO Matrix</li>
#       </ul>
#   </li>
#   <li>Step 3. Putting the DSS Layer Together
#       <ul>
#           <li>The DSS Block</li>
#           <li>Limitations</li>
#       </ul>
#   </li>
# </ul>


# ## Step 1. The Problem with the SSM Convolutional Kernel
#
# We're going to start by taking a step back – back to the original State Space model formulation itself.
#
# ### Rethinking Discretization
# - Sketch SSM as an ODE
# - Motivate need for discretization... how do we discretize? Bilinear method is what S4 uses, but you can also just
# *solve the ODE directly* (yields $\bar{\boldsymbol{A}} = e^{\boldsymbol{A}\Delta}$).
#
# ### Rewriting the SSM Kernel
# - Pull in equation from Part 1 for the kernel.
# - Note repeated multiplication by A (matrix power)
# - Time complexity of matrix power sucks!
# - Unless... *diagonalization*
#
# ### Diagonalization & Efficient Matrix Powers
# - If we can find a way to write $\bar{\boldsymbol{A}}$ as a diagonal matrix, the matrix power defining the kernel
# becomes *trivial*.
# - How?


# ## Step 2. Deriving the Diagonal State Space Model
# Given the benefits of diagonalization, how do we construct a diagonal $\bar{\boldsymbol{A}}$ that leads to efficient
# computation of the SSM kernel $\bar{\boldsymbol{K}}$?
#
# ### Proposition 1 from the DSS Paper
# - Step through original proposition
# - Step through proof in Appendix (simplified)
#
# ### Secret Sauce 1: Complex Softmax
# - Part of the reason this initialization works is because we're initializing our diagonal matrix $\Lambda$ in Complex
# space.
# - This means that our typical softmax stops behaving well... so we need to fix it!
#
# ### Secret Sauce 2: Initializing with the HiPPO Matrix
# - Stability is still tricky
# - HiPPO theory is still necessary (at the beginning) for initializing our weights.


# ## Step 3. Putting the DSS Layer Together
# Mostly just define the DSS Layer and DSSInit function, as well as the final test.
#
# ### Limitations
# - RNN Autoregressive Usage (still being worked out)
# - Still not as performant as S4 in certain settings (expressivity)
# - Still tied to HiPPO theory



## TODO -- Need to weave these parts through the sections above...

def complex_softmax(x, eps=1e-7):
    def reciprocal(x):
        return x.conj() / (x * x.conj() + eps)

    x2 = x - x[np.argmax(x.real)]
    e = np.exp(x2)
    return e * reciprocal(np.sum(e))

def dss_kernel(W, Lambda, L, step):
    P = (step * Lambda)[:, None] * np.arange(L)
    S = jax.vmap(complex_softmax)(P)
    return ((W / Lambda) @ S).ravel().real


def dss_ssm(W, Lambda, L, step):
    N = Lambda.shape[0]
    Abar = np.diag(np.exp(Lambda * step))
    b = jax.vmap(lambda l:
                 1 / (l * (np.exp(l * np.arange(L) * step)).sum()))
    Bbar = b(Lambda).reshape(N, 1)
    Cbar = W.reshape(1, N)
    return Abar, Bbar, Cbar


class DSSLayer(nn.Module):
    Lambda: np.DeviceArray
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # Learned Parameters
        self.W = self.param("W", lecun_normal(), (1, self.N, 2))
        self.W = self.W[..., 0] + 1j * self.W[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(
            self.param("log_step", s4.log_step_initializer(), (1,))
        )
        if not self.decode:
            self.K = dss_kernel(self.W, self.Lambda, self.l_max, self.step)
        else:
            # FLAX code to ensure that we only compute discrete once
            # during decoding.
            def init_discrete():
                return dss_ssm(self.W, self.Lambda, self.l_max, self.step)
            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", np.zeros, (self.N,), np.complex64
            )

    def __call__(self, u):
        if not self.decode:
            return s4.non_circular_convolution(u, self.K) + self.D * u
        else:
            x_k, y_s = s4.scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


DSSLayer = s4.cloneLayer(DSSLayer)


def DSSLayerInit(N):
    _, Lambda, _, _, _ = s4.make_NPLR_HiPPO(2 * N)
    Lambda = Lambda[np.nonzero(Lambda.imag > 0, size=N)]
    return partial(DSSLayer, N=N, Lambda=Lambda)


def test_conversion(N=8, L=16):
    """Maybe this a general test?"""
    step = 1.0 / L
    W = lecun_normal()(rng, (1, N, 2))
    W = W[..., 0] + 1j * W[..., 1]
    _, Lambda, _, _, _ = s4.make_NPLR_HiPPO(2 * N)
    Lambda = Lambda[np.nonzero(Lambda.imag > 0, size=N)]

    K = dss_kernel(W, Lambda, L, step)
    ssm = dss_ssm(W, Lambda, L, step)

    # Apply CNN
    u = np.arange(L) * 1.0
    y1 = s4.non_circular_convolution(u, K.real)

    # Apply RNN
    _, y2 = s4.scan_SSM(
        *ssm, u[:, np.newaxis], np.zeros((N,)).astype(np.complex64)
    )
    assert np.allclose(y1, y2.reshape(-1).real, atol=1e-4, rtol=1e-4)
