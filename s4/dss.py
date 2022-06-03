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
# [follow-up paper applying S4 to audio generation](https://arxiv.org/abs/2202.09729), and weeks later, a [completely
# different group applied S4 to long-range movie clip classification](https://arxiv.org/abs/2204.01692).
#
# Yet, aspects of the implementation remain hard to parse, especially the derivation of the diagonal plus low rank
# constraint on $\boldsymbol{A}$. Not only is this math fairly complex, but in the original PyTorch code base, requires
# the use of custom CUDA kernels -- further obfuscating the implementation (and why this blog uses Jax to efficiently
# compile the relevant operations).
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
#   <li>I. The Problem with the SSM Kernel
#       <ul>
#           <li>Rethinking Discretization</li>
#           <li>Rewriting the SSM Kernel</li>
#           <li>Diagonalization & Efficient Matrix Powers</li>
#       <ul>
#   </li>
#   <li>II. Deriving Diagonal State Spaces
#       <ul>
#           <li>The Annotated Proposition 1</li>
#           <li>Secret Sauce – Part 1: Handling the Complex Softmax</li>
#           <li>Secret Sauce – Part 2: Initializing with HiPPO</li>
#       </ul>
#   </li>
#   <li>III. Putting the DSS Layer Together
#       <ul>
#           <li>The DSS Block</li>
#           <li>Limitations</li>
#       </ul>
#   </li>
# </ul>


# ## I. The Problem with the SSM Kernel
#
# We're going to start by taking a step back – back to the original State Space Model (SSM) itself. The original
# SSM is defined over *continuous* time inputs, as follows (from the original S4 paper)
#
# **[TODO: Link to original post]**

# > The [state space model](https://en.wikipedia.org/wiki/State-space_representation) is defined by this simple equation.
# > It maps a 1-D input signal $u(t)$ to an $N$-D latent state $x(t)$
# > before projecting to a 1-D output signal $y(t)$.
# $$
#   \begin{aligned}
#     x'(t) &= \boldsymbol{A}x(t) + \boldsymbol{B}u(t) \\
#     y(t) &= \boldsymbol{C}x(t) + \boldsymbol{D}u(t)
#   \end{aligned}
# $$
# > Our goal is to simply use the SSM as a black-box representation in a deep
# > sequence model, where $\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C}, \boldsymbol{D}$ are
# > parameters learned by gradient descent...
# >
# > An SSM maps a input $u(t)$ to a state representation vector $x(t)$ and an output $y(t)$.
# > For simplicity, we assume the input and output are one-dimensional, and the state representation
# > is $N$-dimensional. The first equation defines the change in $x(t)$ over time.

# However, when actually training or running inference with this model, we don't take continuous inputs! Instead,
# we usually have a need to *discretize* turning the above differential equation, into a discrete sequence-to-sequence
# map! The key question: how do we discretize?

# ### Rethinking Discretization
#
# One way to discretize the state space model with with the [bilinear method](https://en.wikipedia.org/wiki/Bilinear_transform)
# as described in the original S4 work. This has certain advantages such as **[TODO: advantages of bilinear?]**.
#
# However, a simpler approach to discretizing the SSM is by directly writing each equation in terms of a fixed
# sampling interval $\Delta$, and a discrete index $k$. Doing so results in the following simple system of equations:

# $$
#   \begin{aligned}
#     x((k + 1) \Delta) &= \boldsymbol{\overline{A}}x(k \Delta) + \boldsymbol{\overline{B}} u(k \Delta) \\
#     y(k \Delta) &= \boldsymbol{C}x(k \Delta) + \boldsymbol{D}u(k \Delta)
#   \end{aligned}
# $$

# Solving this system is a simple matter of solving the original ODE and plugging in the results. For solving the
# original SSM equation, [here's a nice reference](https://faculty.washington.edu/chx/teaching/me547/1-7_ss_sol.pdf).
# Then, [this resource provides a nice derivation of the discrete time SSM components](https://users.wpi.edu/~zli11/teaching/rbe595_2017/LectureSlide_PDF/discretization.pdf).

# The punchline of the above derivation is that we can rewrite our SSM -- similar to how we rewrote our SSM for the
# original S4 -- as the following (from the DSS paper):

# > Assuming $A$ is non-singular, for a given sample time $\Delta \in \R_{> 0}$, the discretization of a state space is
# > defined as a sequence-to-sequence map from $(u_0,\ldots,u_{L-1}) = u \in \R^L$ to $(y_0,\ldots,y_{L-1}) = y \in \R^L$
# > where,

# $$
#   \begin{aligned}
#       &x_k = \bar{A}x_{k-1} + \bar{B}u_k\ \ \ ,\ \ \ y_k = \bar{C}x_k  \\[10pt]
#       &\bar{A} = e^{A\Delta}\ \;,\ \bar{B} = (\bar{A} - I)A^{-1}B\ ,\ \;\bar{C} = C\ .
#   \end{aligned}
# $$

# Why is this better than the original parameterization of $\boldsymbol{\overline{A}}$ from the original S4 work? In
# the next section, we'll see how we can derive the SSM kernel using this parameterization with simpler restrictions on
# the structure of $\boldsymbol{\overline{A}}$, allowing for a *simple, straightforward* implementation without losing
# much in the way of performance!

# ### Rewriting the SSM Kernel
#
# **[TODO figure out cross-page links]**
#
# Part 1 of this post showed that the above discretized state-space model can be treated as a *sequence-to-sequence* map,
# behaving a lot like an RNN with a transition matrix given by $\boldsymbol{\overline{A}}$:

# $$
# \begin{aligned}
#   x_{k} &= \boldsymbol{\overline{A}} x_{k-1} + \boldsymbol{\overline{B}} u_k\\
#   y_k &= \boldsymbol{\overline{C}} x_k \\
# \end{aligned}
# $$

# We then showed how we can turn the above recurrence into a *convolution* given the repetitive structure! We end up with
# the kernel:

# $$
# \begin{aligned}
#     y_k &= \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^k \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^{k-1} \boldsymbol{\overline{B}} u_1 + \dots + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_{k-1} + \boldsymbol{\overline{C}}\boldsymbol{\overline{B}} u_k
#     \\
#     y &= \boldsymbol{\overline{K}} \ast u
# \end{aligned}
# $$

# $$
# \begin{aligned}
#   \boldsymbol{\overline{K}} \in \mathbb{R}^L  = (\boldsymbol{\overline{C}}\boldsymbol{\overline{B}}, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}\boldsymbol{\overline{B}}, \dots, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}^{L-1}\boldsymbol{\overline{B}})
# \end{aligned}
# $$

# **Problem**: Unfortunately, just like with the original S4 paper, computing this kernel $\boldsymbol{\overline{K}}$ is
# **prohibitively expensive** (successive matrix powers of $\boldsymbol{\overline{A}}$ which blows up assuming
# $\mathcal{O}(d^3)$ matrix multiplication, where $d$ is the dimensionality of $\boldsymbol{\overline{A}}$). Getting SSMs
# to scale requires finding an *alternative path* to computing this kernel – one that is both efficient and that doesn't
# badly restrict the expressivity of $\boldsymbol{\overline{A}}$. So how can we address this?


# ### Diagonalization & Efficient Matrix Powers

# This is the key "fork in the road" between the original S4 paper, and this post's Diagonal State Spaces paper. Notably,
# where the S4 paper is rooted in HiPPO theory and steps through some complex math (and complex code!) to make computing
# the kernel $\boldsymbol{\overline{K}}$ efficient, the DSS is going to make a single assumption: let
# $\boldsymbol{\overline{A}}$ be *diagonalizable*.
#
# Doing so turns an expensive $\mathcal{O}(d^3)$ matrix multiply into a near-linear time operation, one that is
# conducive to performing matrix powers super fast! How we can write and initialize $\boldsymbol{\overline{A}}$ in
# this way, and produce an update rule that ensure stable learning is the focus of the next section.

# ## Part II. Deriving Diagonal State Spaces
#
# As a brief sketch, the DSS paper shows  that we simply need to break $\boldsymbol{\overline{A}}$ into a collection
# of diagonal terms $\Lambda = \lambda_1 \ldots \lambda_n$; then with some straightforward algebra, we can compute an
# efficient expression for our kernel $\boldsymbol{\overline{K}}$.
#
# We present this derivation (effectively Proposition 1 of the DSS paper) with light annotation below.

# ### The Annotated Proposition 1
#
# Recall our expanded kernel $\boldsymbol{\overline{K}}$:

# $$
# \begin{aligned}
#   \boldsymbol{\overline{K}} \in \mathbb{R}^L  = (\boldsymbol{\overline{C}}\boldsymbol{\overline{B}}, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}\boldsymbol{\overline{B}}, \dots, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}^{L-1}\boldsymbol{\overline{B}})
# \end{aligned}
# $$

# Proposition 1 defines an efficient expression for computing $\boldsymbol{\overline{K}}$:
#
# > **Proposition:** Let $\boldsymbol{\overline{K}} \in \R^{1\times L}$ be the kernel for a state space
# > $(\boldsymbol{\overline{A}}, \boldsymbol{\overline{B}}, \boldsymbol{\overline{C}})$ and sample time $\Delta > 0$.
#
# > If $\boldsymbol{\overline{A}} \in \mathbb{C}^{N \times N}$ is diagonalizable over $\mathbb{C}$ with eigenvalues
# > $\lambda_1,\ldots,\lambda_N$ such that, $\forall i$, $\lambda_i \neq 0$ and $e^{L\lambda_i\Delta} \neq 1$,
# > then $\exists w \in \mathbb{C}^{1 \times N}$ such that:$\\[2pt]$
# $$
# \begin{aligned}
#   \bar{K} = w \cdot \Lambda^{-1} \cdot \mathrm{row}{\text -}\mathrm{softmax}(P_{N\times L})
# \end{aligned}
# $$
# > where $P_{i,k} = \lambda_i k\Delta$, and $\Lambda$ is diagonal matrix of $\lambda_1,\ldots,\lambda_N$.

# Plainly, there are three parts to this proposition:
#   1. Given we can diagonalize $\boldsymbol{\overline{A}}$, we'll store its diagonal components $\lambda_1 \ldots
#   \lambda_n$ in $\Lambda$.
#   2. The learned term $w$ is going to store some aggregate information of our other state space matrices
#   $\boldsymbol{\overline{B}}$, $\boldsymbol{\overline{C}}$. We'll show how this happens in the proof below.
#   3. Finally, given this particular structure of $\boldsymbol{\overline{A}}$, we can write the full kernel
#   $\boldsymbol{\overline{K}}$ as the product of the inverse of $\Lambda$, this aggregate term $w$, as well as a
#   separate softmax term $P$ that encodes some sequence positional information, blended with our diagonal terms
#   $\Lambda$.

# Put another way – working out the math for the DSS formulation of the state space model **lets us write the kernel
# as a simple product of some diagonal terms, a learned vector $w$, and a easy-to-formulate position matrix $P$**.
#
# Let's derive this!

# > **Proof:** Let $A$ be diagonalizable over $\mathbb{C}$ as $A = V \Lambda V^{-1}$ with eigenvalues
# > $\lambda_1,\ldots, \lambda_N \in \mathbb{C}$. From the above expression of the SSM kernel we have:
# $$
# \begin{aligned}
#   \boldsymbol{\overline{K}} &= (\boldsymbol{\overline{C}}\boldsymbol{\overline{B}},
#   \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}\boldsymbol{\overline{B}},\ldots,
#   \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}^{L-1}\boldsymbol{\overline{B}})
# \end{aligned}
# $$
# > where,
# $$
# \begin{aligned}
#   \boldsymbol{\overline{K}}_k &= \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}^k\boldsymbol{\overline{B}}
#   = C e^{A\cdot k\Delta} (e^{A\Delta} - I)A^{-1}B \\
#   &= (CV) e^{\Lambda k\Delta}(e^{\Lambda\Delta} - I)\Lambda^{-1} (V^{-1}B)
# \end{aligned}
# $$
# > For $CV \in \mathbb{C}^{1 \times N}$ and $V^{-1}B \in \mathbb{C}^{N \times 1}$ let
# > $(CV)^\top * (V^{-1}B) = \widetilde{w} \in \mathbb{C}^N$ be the element-wise product of $CV$ and $V^{-1}B$. Then,
# $$
# \begin{aligned}
#   \boldsymbol{\overline{K}}_k &= \sum_{i=1}^N {e^{\lambda_i k\Delta}(e^{\lambda_i\Delta} - 1) \over \lambda_i} \cdot \widetilde{w}_i \\[2pt]
#   &= \sum_{i=1}^N {e^{\lambda_i k\Delta}(e^{\lambda_i\Delta} - 1) \over \lambda_i(e^{L\lambda_i\Delta} - 1)} \cdot (\widetilde{w}_i \cdot (e^{L\lambda_i\Delta} - 1)) \\[2pt]
#   &= \sum_{i=1}^N (\widetilde{w}_i \cdot (e^{L\lambda_i\Delta} - 1))\cdot \frac{1}{\lambda_i} \cdot {e^{\lambda_i k\Delta} \over \sum_{r=0}^{L-1} e^{r\lambda_i\Delta}}
# \end{aligned}
# $$
# > where the last equality follows from $(z^L-1) = (z-1)(z^0+\ldots+z^{L-1})$ and using $z^L \neq 1$.
# >
# > Let $P \in \mathbb{C}^{N \times L}$ be the matrix $P_{i,k} = \lambda_i \cdot k\Delta$ and
# > $S = \mathrm{row}{\text -}\mathrm{softmax}(P)$ denote the matrix obtained after applying $\mathrm{softmax}$ on
# > the rows of $P$, i.e.
# $$
# \begin{aligned}
#   S_{i,k} = {e^{\lambda_i k\Delta} \over \sum_{r=0}^{L-1} e^{r\lambda_i\Delta}}
# \end{aligned}
# $$
# >
# > Let $w \in \mathbb{C}^N$ be defined as $$w_i = \widetilde{w}_i \cdot (e^{L\lambda_i\Delta} - 1).$$
# >
# > Then, plugging in each of the above definitions into the expression for \boldsymbol{\overline{K}}_k above, we get:
# $$
# \begin{aligned}
#   \boldsymbol{\overline{K}}_k &= \sum_{i=1}^N (\widetilde{w}_i \cdot (e^{L\lambda_i\Delta} - 1))\cdot \frac{1}{\lambda_i} \cdot {e^{\lambda_i k\Delta} \over \sum_{r=0}^{L-1} e^{r\lambda_i\Delta}} \\[2pt]
#   &= \sum_{i=1}^N w_i \cdot \frac{1}{\lambda_i} \cdot S_{i, k} \\[2pt]
#   &= w \cdot \Lambda^{-1} \cdot \mathrm{row}{\text -}\mathrm{softmax}(P_{N\times L})
# \end{aligned}
# $$
# > completing the proof.

# Computing the kernel in this way (collapsing the $\boldsymbol{\overline{B}}$ and $\boldsymbol{\overline{C}}$ terms into
# $w$ has advantages for the complexity of computing the kernel and running the discrete convolution. Namely,
# > For batch size $B$, sequence length $L$ and hidden size $H$, the DSS layer requires $O(NHL)$ time and space to
# > compute the kernels, $O(BHL\log(L))$ time for the discrete convolution and $O(BH^2L)$ time for the output projection.

# More importantly, implementing the DSS kernel is *very* straightforward:

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
    b = jax.vmap(lambda l: 1 / (l * (np.exp(l * np.arange(L) * step)).sum()))
    Bbar = b(Lambda).reshape(N, 1)
    Cbar = W.reshape(1, N)
    return Abar, Bbar, Cbar


def test_conversion(N=8, L=16):
    """Test the equivalence of the DSS kernel with the generic SSM kernel."""
    step = 1.0 / L
    W = lecun_normal()(rng, (1, N, 2))
    W = W[..., 0] + 1j * W[..., 1]
    Lambda, _, _, _ = s4.make_DPLR_HiPPO(N)

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


test_conversion()


# ### Secret Sauce – Part 1: Complex Softmax
#
# While the implementation above is pretty concise, there are some subtle gotchas that need to be addresed. First is the
# computation of the special $\mathrm{row}{\text -}\mathrm{softmax}()$ function.
#
# Note that with the given derivation, many of the state space matrices are defined over *complex* space! The
# traditional softmax function we've come to know and love has some problems operating in complex space – for example,
# consider the complex $\mathrm{softmax}(0, \pi i)$; taking the naive softmax results in division by zero,
# as the denominator is $e^{0} + e^{\pi i} = 1 - 1 = 0$!
#
# To correct for this, the DSS paper defines a slight correction to the softmax function, to ensure stability:
#
# > As noted above, $\mathrm{softmax}$ can have singularities over $\mathbb{C}$. To address this issue, we use a simple
# > correction to make it well-defined over the entire domain:
# >
# > $\mathrm{softmax}$:
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Given $(x_0,\ldots,x_{L-1}) = x \in \mathbb{C}^L$,
# > let $\mathrm{softmax}(x) \in \mathbb{C}^L$ be defined as:
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(\mathrm{softmax}(x))_k = e^{x_k} (e^{x_0} + \ldots +
# e^{x_{L-1}})^{-1}.$
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Note that for any $c \in \mathbb{C}$, $\mathrm{softmax}(x_0,\ldots,x_{L-1})$ $=$
#   $\mathrm{softmax}(x_0-c,\ldots,x_{L-1}-c)$.
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Unlike over $\mathbb{R}$, $\mathrm{softmax}$ can have singularities over $\mathbb{C}$ as sum of
#   exponentials can vanish.
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;E.g. $e^{0} + e^{i\pi} = 0$ and hence $\mathrm{softmax}(0,i\pi)$ is not defined.
# >
# > $\mathrm{max}$:
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Given $(x_0,\ldots,x_{L-1}) = x \in \mathbb{C}^L$, let
#   $\mathrm{max}(x)$ be the $x_i$ with the maximum real part,
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i.e. $x_{\mathrm{argmax}_i \mathrm{Re}(x_i)}$.
# >
# > $\mathrm{reciprocal}_\epsilon$:
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Given $x \in \mathbb{C}$ and $\epsilon \in \R_{> 0}$, let
#   $\mathrm{reciprocal}_\epsilon(x) = \frac{\overline{x}}{x\cdot \overline{x} + \epsilon}$ where $\overline{x}$ is
#   the complex conjugate of $x$.
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The denominator is always in $\R_{\geq \epsilon}$ and
#   $|\mathrm{reciprocal}_\epsilon| \leq (2\sqrt{\epsilon})^{-1}$.
# >
# > $\mathrm{softmax}_\epsilon$:
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Given $(x_0,\ldots,x_{L-1}) = x \in \mathbb{C}^L$ let
#   $m = \mathrm{max}(x)$ and $\widetilde{x}_i = x_i - m$. Note that $|e^{\widetilde{x}_i}| \leq 1$.
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Given $\epsilon \in \R_{> 0}$, let $\mathrm{softmax}_\epsilon(x)
#   \in \mathbb{C}^L$ be:
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$(\mathrm{softmax}_\epsilon(x))_k = e^{\widetilde{x}_k}
#   \cdot\mathrm{reciprocal}_\epsilon\left(\sum_{r=0}^{L-1}  e^{\widetilde{x}_r}\right)$$
# >
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathrm{softmax}_\epsilon$ is always bounded and differentiable.
# >
# > In the DSS implementation, we use $\mathrm{softmax}_\epsilon$ with $\epsilon = 10^{-7}$.

# As a punchline – to stabilize the softmax to work over $\mathbb{C}$, we write a new $\mathrm{softmax}_\epsilon$
# where we first adjust each element by subtracting out the max real component, then reformulate the reciprocal
# (denominator) in the traditional computation to always output a real number (by multiplying by the complex conjugate).
#
# ### Secret Sauce – Part 2: Initializing with the HiPPO Matrix
#
# One other sticking point you might notice in the above code is in *how we initialize the diagonal values $\Lambda$. In
# order to ensure stability during training, we *must* initialize our $\Lambda$ subject to the HiPPO initialization from
# the S4 paper and prior work.
#
# The reasoning for this is mostly due to stability; repeated matrix powers of $\boldsymbol{\overline{A}}$ still need to
# be of low condition number such that the kernel doesn't explode. HiPPO theory gives us a solid grounding and a
# reasonable initialization to use, at minimal cost (it's a fixed initialization to use at the beginning of training!).


# ## Step III. Putting the DSS Layer Together
#
# Now that we've defined all the requisite pieces – the simplified expression for the kernel
# $\boldsymbol{\overline{K}}$, the corrected $\mathrm{softmax}$ function, and the initialization for $\Lambda$,
# we're ready to put the DSS layer together!


class DSSLayer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # Learned Parameters
        hippo_Lambda_real_initializer, hippo_Lambda_imag_initializer, hippo_p_initializer, hippo_B_initializer = s4.hippo_initializer(self.N)
        self.Lambda_re = self.param("Lambda_re", hippo_Lambda_real_initializer, (self.N,))
        self.Lambda_im = self.param("Lambda_im", hippo_Lambda_imag_initializer, (self.N,))
        self.Lambda = self.Lambda_re + 1j*self.Lambda_im
        self.W = self.param("W", lecun_normal(), (1, self.N, 2))
        self.W = self.W[..., 0] + 1j * self.W[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(
            self.param("log_step", s4.log_step_initializer(), (1,))
        )
        if not self.decode:
            self.K = dss_kernel(self.W, self.Lambda, self.l_max, self.step)
        else:
            # FLAX code to ensure that we only compute discrete once during decoding.
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

def DSSLayerInit(N):
    return partial(DSSLayer, N=N)


DSSLayer = s4.cloneLayer(DSSLayer)


# The core of the DSS layer is the same as the traditional SSM layer defined in the first part of the post. We define
# the initializer, define our learnable weights $w$ then call the kernel code written above as a convolution during
# training.
#
# Finally, during discrete decoding, we use the initial recurrence computed above.
#
# ... and that's all folks! DSS is not only more compact (< 100 LoC) than S4, but at it's core is a simple idea:
# diagonalization allows one to efficiently compute matrix powers, and we can use that insight to build a kernel that
# is almost as expressive and just as performant as S4.
