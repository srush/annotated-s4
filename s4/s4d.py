# <center>
# <h1>
# The Annotated DSS / S4D:  
# Diagonal State Space Models
# </h1>
# </center>

#
# <center>
# <p><a href="https://arxiv.org/abs/2203.14343">Diagonal State Spaces are as Effective as Structured State Spaces</a></p>
# <p> Ankit Gupta, Albert Gu, Jonathan Berant.</p>
# </center>
#
# <center>
# <p><a href="https://TODO">On the Parameterization and Initialization of Diagonal State Space Models</a></p>
# <p> Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré.</p>
# </center>
# <img src="images/s4d.png" width="100%"/>
#
# ---
#
# *Note: This page is meant as a standalone complement to Section 2 [TODO Link] of the original
# blog post.*
#
# The months following the release of the S4 paper by Gu et al. were characterized by a wave of excitement around the new
# model, it's ability to handle extremely long sequences, and generally, what such a departure from Transformer-based
# architectures could mean. The original authors came out with a
# [follow-up paper applying S4 to audio generation](https://arxiv.org/abs/2202.09729), and weeks later, a [completely
# different group applied S4 to long-range movie clip classification](https://arxiv.org/abs/2204.01692).

# Yet, S4 has an intricate algorithm that requires a complicated implementation for **diagonal plus low rank** (DPLR) state space models (SSM).
# To motivate this representation, S4 considered the case of **diagonal** state matrices,
# and outlined a simple method that can be implemented in just a few lines.
# However, this was not used because no diagonal SSMs were known that could mathematically model long-range dependencies - S4's ultimate goal.
# Instead, S4 used a class of special matrices that could not be diagonalized, but found a way to transform them into *almost diagonal* form,
# leading to the more general DPLR representation.

# However, at the end of March 2022 - an effective diagonal model was discovered in [[Diagonal State Spaces are as Effective as Structured State Spaces](https://arxiv.org/abs/2203.14343)] based on approximating S4's matrix (DSS).
# This important observation allowed diagonal SSMs to be used while preserving the empirical strengths of S4!
# Diagonal SSMs were further fleshed out in [[On the Parameterization and Initialization of Diagonal State Space Models](https://TODO)],
# which used an even simpler method based on S4's original outline, combined with new theory explaining why DSS's diagonal initialization can model long-range dependencies (S4D).
# The rest of this post steps through the incredibly simple model and theoretical intuition of S4D, an *even more structured* state space.
#
# This post aims to be **a complete standalone for Section 2** of the original Annotated S4 post.
# We'll still be using Jax with the Flax NN Library for consistency with the original post, and PyTorch versions of [DSS](https://github.com/ag1988/dss) and [S4D](https://github.com/HazyResearch/state-spaces) models are publically available.

from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from .s4 import (
    causal_convolution,
    cloneLayer,
    hippo_initializer,
    log_step_initializer,
    make_DPLR_HiPPO,
    scan_SSM,
)


if __name__ == '__main__':
    rng = jax.random.PRNGKey(1)


# ## Table of Contents
# <nav id="TOC">
# * [Part I. A Refresher on State Space Models]
# * [Part II. Diagonal State Space Models]
#     - [The Diagonal SSM Kernel: Vandermonde Matrix Multiplication]
#     - [Implementing the S4D Kernel]
#     - [Comparing SSM Parameterizations and Efficiency]
#     - [Computational Complexities]
#     - [The Complete S4D Layer]
# * [Part IIIa. The Central Challenge: Initialization]
#     - [A Brief Refresher on S4 and HiPPO]
#     - [The Diagonal HiPPO Matrix]
# * [Part IIIb. An Intuitive Understanding of SSMs]
# </nav>
# <!--
#     - [Case: 1-dimensional State]
#     - [Case: Diagonal SSM]
#     - [Case: General SSM]
#     - [Case: HiPPO and Diagonal-HiPPO]
#     - [Other Diagonal Initializations]
# -->


# Part I of this post provides a quick summary of SSMs to define their main computational challenge.
# In Part II, we step through the complete derivation and implementation of S4D, following the original S4 paper.
# Notably, the core kernel computation requires **only 2 lines of code**!
# Finally, Part III covers the theory of diagonal SSMs, from how S4 originally modeled long-range dependencies, to the new breakthroughs in initializing DSS and S4D.


# ## Part I. A Refresher on State Space Models
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
#     y(t) &= \boldsymbol{C}x(t)
#   \end{aligned}
# $$
# > Our goal is to simply use the SSM as a black-box representation in a deep
# > sequence model, where $\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C}$ are
# > parameters learned by gradient descent...
# >
# > An SSM maps a input $u(t)$ to a state representation vector $x(t)$ and an output $y(t)$.
# > For simplicity, we assume the input and output are one-dimensional, and the state representation
# > is $N$-dimensional. The first equation defines the change in $x(t)$ over time.

# [AG: In the DSS post, Sidd's elaboration on discretization is great and should be in Part 1 of the Annotated S4, as they are general facts about SSMs independent of S4/DSS. I also recommend looking at my [blog post on discretization](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-3)]

# Recall also that in discrete time, the SSM is viewed as a sequence-to-sequence map $(u_k) \mapsto (y_k)$,
# where the sequence $u_k = u(k \Delta)$ represents sampling the underlying continuous $u(t)$ with a fixed sampling interval or step size $\Delta$.

#
# Part 1 of the S4 post showed that this discretized state-space model can be viewed as a linear RNN
# with a transition matrix given by $\boldsymbol{\overline{A}}$:

# $$
# \begin{aligned}
#   x_{k} &= \boldsymbol{\overline{A}} x_{k-1} + \boldsymbol{\overline{B}} u_k\\
#   y_k &= \boldsymbol{\overline{C}} x_k \\
# \end{aligned}
# $$

# Note that when $\boldsymbol{A}$ is diagonal, the first equation decomposes as independent 1-dimensional recurrences over the elements of $x$ (*Splash figure, Left*)!

# We then showed how we can turn the above recurrence into a *convolution* because of the repetitive structure (more formally because the recurrence is *time-invariant*).
# Expanding out the recurrence gives a closed formula for $y$
# $$
# \begin{aligned}
#     y_k &= \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^k \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^{k-1} \boldsymbol{\overline{B}} u_1 + \dots + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_{k-1} + \boldsymbol{\overline{C}}\boldsymbol{\overline{B}} u_k
# \end{aligned}
# $$
# which is just a convolution with a particular kernel $\bm{\overline{K}}$:
# $$
# \begin{aligned}
#     y &= \boldsymbol{\overline{K}} \ast u \\
#   \boldsymbol{\overline{K}} &= (\boldsymbol{\overline{C}}\boldsymbol{\overline{B}}, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}\boldsymbol{\overline{B}}, \dots, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}^{L-1}\boldsymbol{\overline{B}}) \in \mathbb{R}^L
# \end{aligned}
# $$

# Recall that $N$ denotes the state size, or the dimensionality of $\boldsymbol{A} \in \mathbb{C}^{N \times N}, \boldsymbol{B} \in \mathbb{C}^{N \times 1}, \boldsymbol{C} \in \mathbb{C}^{1 \times N}$, while $L$ denotes the sequence length.

# $$
# \begin{aligned}
#   \boldsymbol{\overline{K}} \in \mathbb{R}^L  = (\boldsymbol{\overline{C}}\boldsymbol{\overline{B}}, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}\boldsymbol{\overline{B}}, \dots, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}^{L-1}\boldsymbol{\overline{B}})
# \end{aligned}
# $$

# **Problem**: SSMs in deep learning have two core challenges. The *modeling* challenge is finding good parameters of the SSM, particular the state matrix $\boldsymbol{A}$, that can effectively model complex interactions in sequential data such as long-range dependencies. We defer this discussion, which is more theoretical, to Part III.
#
# The core *computational* challenge of SSMs is constructing this kernel $\boldsymbol{\overline{K}}$ fast. Overcoming this requires imposing *structure* on the state space. Next, we'll see how diagonal SSMs provide a simple way to do this.

# ## Part II. Diagonal State Space Models

# Let's now examine more closely how to compute the discretized SSM kernel.
# This part will directly follow Section 3.1 of the original S4 paper.

# > The fundamental bottleneck in computing the discrete-time SSM is that it involves repeated matrix multiplication by $\boldsymbol{\overline{A}}$.
# > For example, computing $\boldsymbol{\overline{K}}$ naively involves $L$ successive multiplications by $\boldsymbol{\overline{A}}$, requiring $O(N^2 L)$ operations and $O(NL)$ space.

# In other words, computing this kernel $\boldsymbol{\overline{K}}$ can be
# prohibitively expensive for general state matrices $\boldsymbol{A}$, which was an issue in the [predecessor to S4](https://arxiv.org/abs/2110.13985). Getting SSMs
# to scale requires finding an alternative way to computing this kernel – one that is both efficient and doesn't
# badly restrict the expressivity of $\boldsymbol{A}$. How can we address this?

# > To overcome this bottleneck, we use a structural result that allows us to simplify SSMs.
# >
# > **Lemma 1.**
# >   Conjugation is an equivalence relation on SSMs
# $$
# (\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C}) \sim (\boldsymbol{V}^{-1} \boldsymbol{A} \boldsymbol{V}, \boldsymbol{V}^{-1}\boldsymbol{B}, \boldsymbol{C}\boldsymbol{V})
# $$
# >
# > **Proof.**
# > Write out the two SSMs with state denoted by $x$ and $\tilde{x}$ respectively:
# > $$
# > \begin{aligned}
# >   x' &= \boldsymbol{A}x + \boldsymbol{B}u & \qquad \qquad \qquad \tilde{x}' &= \boldsymbol{V}^{-1}\boldsymbol{A}\boldsymbol{V}\tilde{x} + \boldsymbol{V}^{-1}\boldsymbol{B}u \\
# >   y &= \boldsymbol{C}x & \qquad \qquad \qquad y &= \boldsymbol{C}\boldsymbol{V}\tilde{x}
# > \end{aligned}
# > $$
# > After multiplying the right side SSM by $\boldsymbol{V}$, the two SSMs become identical with $x = \boldsymbol{V}\tilde{x}$.
# > Therefore these compute the exact same operator $u \mapsto y$, but with a change of basis by $\boldsymbol{V}$ in the state $x$.

# Why is this important? It allows replacing $\boldsymbol{A}$ with a [canonical form](https://en.wikipedia.org/wiki/Canonical_form#Linear_algebra) such as diagonal matrices,
# imposing simpler *structure* while preserving expressivity! Ideally, this structure would simplify and speed up the computation of the SSM kernel.
#
# Note that Lemma 1 provides an immediately implies the expressivity of diagonal SSMs.
# To spell it out: suppose we have a state space with parameters $(\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C})$ where the matrix $\boldsymbol{A}$ is diagonalizable - in other words, there exists a matrix $\boldsymbol{V}$ such that $\boldsymbol{V}^{-1}\boldsymbol{A}\boldsymbol{V}$ is diagonal.
# Then the state space $(\boldsymbol{V}^{-1} \boldsymbol{A} \boldsymbol{V}, \boldsymbol{V}^{-1}\boldsymbol{B}, \boldsymbol{C}\boldsymbol{V})$ is a diagonal SSM that is *exactly equivalent*, or in other words defines the exact same sequence-to-sequence transformation $u \mapsto y$!

# Furthermore, it's well known that [almost all square matrices are diagonalizable](https://chiasme.wordpress.com/2013/09/03/almost-all-matrices-are-diagonalizable/), so that diagonal SSMs are essentially fully expressive (with a caveat that we'll talk about in Part III).

# **Remark.** Note that Lemma 1 is about equivalence of *continuous* SSMs. The equivalence of their discretizations follows immediately because the *discrete* SSM (viewed as the map $u_k \mapsto y_k$) depends only on the step size $\Delta$ and the continuous SSM (as the map $u(t) \mapsto y(t)$). A longer version of this expressivity result is presented as Proposition 1 of the DSS paper, which focuses on the discrete case.]

#
# [**AG:** Not sure if the above remark is useful or just distracting]

# ### The Diagonal SSM Kernel: Vandermonde Matrix Multiplication

# So what's the computational advantage of diagonal SSMs? S4 outlined the main idea:

# > Lemma 1 motivates putting $\bm{A}$ into a canonical form by conjugation, which is ideally more structured and allows faster computation.
# > For example, if $\bm{A}$ were diagonal, the resulting computations become much more tractable.
# > In particular, the desired $\bm{\overline{K}}$ would be a **Vandermonde product** which theoretically only needs $O((N+L)\log^2(N+L))$ arithmetic operations.

# Let's elaborate.
# The key idea is that when $\boldsymbol{\overline{A}}$ is diagonal, the matrix power can be broken into a collection of *scalar* powers,
# dramatically simplifying the structure of the kernel $\boldsymbol{\overline{K}}$.

# Notationally, recall that $\boldsymbol{\overline{A}} \in \mathbb{C}^{N \times N}, \boldsymbol{\overline{B}} \in \mathbb{C}^{N \times 1}, \boldsymbol{C} \in \mathbb{C}^{1 \times N}$. When $\boldsymbol{\overline{A}}$ is diagonal, we'll slightly overload notation to let $\boldsymbol{\overline{A}}_i, \boldsymbol{\overline{B}}_i, \boldsymbol{C}_i$ denote their scalar entries for simplicity.

# So the $\ell$-th element of the convolution kernel is (a scalar)
# $$
# \boldsymbol{\overline{K}}_\ell = \boldsymbol{C}\boldsymbol{\overline{A}}^\ell\boldsymbol{\overline{B}} = \sum_{n=0}^{N-1} \boldsymbol{C}_n \boldsymbol{\overline{A}}_n^\ell \boldsymbol{\overline{B}}_n
# $$

# But this can be rewritten as a single matrix-vector product,
# where the matrix on the right side is known as a [Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix), whose columns encode successive powers of $\boldsymbol{\overline{A}}$.

# $$
# \begin{aligned}
#       \boldsymbol{\overline{K}} =
#       \begin{bmatrix}
#         \boldsymbol{C}_0 \boldsymbol{\overline{B}}_0 & \dots & \boldsymbol{C}_{N-1} \boldsymbol{\overline{B}}_{N-1}
#       \end{bmatrix}
#       \begin{bmatrix}
#         1      & \boldsymbol{\overline{A}}_0     & \boldsymbol{\overline{A}}_0^2     & \dots  & \boldsymbol{\overline{A}}_0^{L-1}     \\
#         1      & \boldsymbol{\overline{A}}_1     & \boldsymbol{\overline{A}}_1^2     & \dots  & \boldsymbol{\overline{A}}_1^{L-1}     \\
#         \vdots & \vdots                  & \vdots                    & \ddots & \vdots                        \\
#         1      & \boldsymbol{\overline{A}}_{N-1} & \boldsymbol{\overline{A}}_{N-1}^2 & \dots  & \boldsymbol{\overline{A}}_{N-1}^{L-1} \\
#       \end{bmatrix}
#     \end{aligned}
# $$


# More importantly, writing the kernel in this form immediately exposes the computational complexity!
# Naively, materializing the $N \times L$ matrix requires $O(NL)$ space and the multiplication takes $O(NL)$ time.
# But Vandermonde matrices are very well-studied, and it's known that they can be multiplied in $\widetilde{O}(N+L)$ operations and $O(N+L)$ space,
# providing a theoretical asymptotic efficiency improvement.

# In practice, our implementation below will use the naive $O(NL)$ summation but leverage the structure of the Vandermonde matrix to avoid materializing it, reducing the space complexity to $O(N+L)$.
#  The main idea is that the Vandermonde matrix has a simple formula in terms of its parameters $\boldsymbol{A}$, so its entries can be computed on demand instead of all in advance. For example, computing each $\boldsymbol{\overline{K}}_\ell$ one by one, materializing one column of the matrix at a time, would be much more memory efficient. In JAX, this can be automatically handled by JIT and XLA compilation.
# This is a nice sweet spot that's simple to implement, memory efficient, and quite fast on modern parallelizable hardware like GPUs and TPUs.
# We'll comment more on the efficiency in [[Comparing SSM Parameterizations and Efficiency]].

# We also make note of another small implementation detail: from the above formula, diagonal SSMs depends only on the elementwise product $\boldsymbol{C} \circ \boldsymbol{B}$.
# So we can assume without loss of generality that $\boldsymbol{B} = \boldsymbol{1}$ and choose to either train it (as in S4(D)) or freeze it (as in DSS).
# <!--
# ^[Instead of the notation $\bm{B}$ and $\bm{C}$, DSS defines a $\bm{W}$ parameter which represents $\boldsymbol{C} \circ \boldsymbol{B}$. This is equivalent to setting $\bm{B} = \bm{1}$ and freezing it, while S4D chooses to train it in the style of the original S4.]
# -->

# ### Implementing the S4D Kernel

# Implementing this simple version of S4 for the diagonal case is very straightforward.
# As with all SSMs, the first step is to discretize the parameters with a step size $\Delta$. [**Link to Post 1**]
# This is much simpler for diagonal state matrices $\boldsymbol{A}$, as the discretizations normally involves matrix inverses or exponentials that can be broken up into scalar operations.

def discretize(A, B, step, mode="zoh"):
    if mode == "bilinear":
        num, denom = 1 + .5 * step*A, 1 - .5 * step*A
        return num / denom, step * B / denom
    elif mode == "zoh":
        return np.exp(step*A), (np.exp(step*A)-1)/A * B

# Here we show both the Bilinear method used in S4 and HiPPO, and the ZOH method used in other SSMs such as DSS and [LMU](https://papers.nips.cc/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html).
# (As discussed in Part 1 of the Annotated S4 [AG: if we put more about discretization there], these are closely related and have no real empirical difference.)

# As described in the original paper, the kernel in the diagonal case is just a single **Vandermonde matrix-vector product**. This is almost trivial to implement and can be applied to *any discretization* of a diagonal SSM.

def vandermonde_product(v, alpha, L):
    V = alpha[:, np.newaxis] ** np.arange(L)  # Vandermonde matrix
    return (v[np.newaxis, :] @ V)[0]


def s4d_kernel(C, A, L, step):
    Abar, Bbar = discretize(A, 1.0, step)
    return vandermonde_product(C * Bbar, Abar, L).real


# Finally, this kernel can be slightly optimized.
# First, computing powers $\alpha^k$ explicitly can be slower than exponentiating $\exp(k \log(\alpha))$.
# Second, in the case of ZOH discretization (which directly involves a matrix exponential), a $\log \circ \exp$ term can be removed, saving a pointwise operation.
# Finally, as mentioned above, materializing the full matrix is unnecessary and can be optimized away to save a lot of memory! We simply write the kernel in a way that exposes the structure (via `vmap`) and let JAX and XLA handle the rest.


@partial(jax.jit, static_argnums=2)
def s4d_kernel_zoh(C, A, L, step):
    kernel_l = lambda l: (C * (np.exp(step*A)-1)/A * np.exp(l*step*A)).sum()
    return jax.vmap(kernel_l)(np.arange(L)).real


# We highlight that the above *2 lines of code* is a drop-in replacement for all the intricate machinery of the full S4 model!
#
# Just as with all SSMs, we can test that convolving by this kernel produces the same answer as the sequential scan.


def s4d_ssm(C, A, L, step):
    N = A.shape[0]
    Abar, Bbar = discretize(A, np.ones(N), step, mode="zoh")
    Abar = np.diag(Abar)
    Bbar = Bbar.reshape(N, 1)
    Cbar = C.reshape(1, N)
    return Abar, Bbar, Cbar


def test_conversion(N=8, L=16):
    """Test the equivalence of the S4D kernel with the generic SSM kernel."""
    step = 1.0 / L
    C = normal()(rng, (N, 2))
    C = C[..., 0] + 1j * C[..., 1]
    A, _, _, _ = make_DPLR_HiPPO(N)
    A = A[np.nonzero(A.imag > 0, size=N)]

    K_ = s4d_kernel(C, A, L, step)
    K = s4d_kernel_zoh(C, A, L, step)
    assert np.allclose(K_, K, atol=1e-4, rtol=1e-4)

    ssm = s4d_ssm(C, A, L, step)

    # # Apply CNN
    u = np.arange(L) * 1.0
    y1 = causal_convolution(u, K)

    # # Apply RNN
    _, y2 = scan_SSM(
        *ssm, u[:, np.newaxis], np.zeros((N,)).astype(np.complex64)
    )
    assert np.allclose(y1, y2.reshape(-1).real, atol=1e-4, rtol=1e-4)

if __name__ == '__main__':
    test_conversion()

# ### Comparing SSM Parameterizations and Efficiency

# With all these different SSM methods floating around, let's quickly compare some versions of SSMs to understand their similarities and differences, historical context, and computational complexities.

# **S4.**
# First, let's revisit once more the main point of S4's algorithm, which dramatically improved the efficiency of computing the SSM kernel for DPLR matrices.

# > For state dimension $N$ and sequence length $L$, computing the latent state requires $O(N^2 L)$ operations and $O(NL)$ space - compared to a $\Omega(L+N)$ lower bound for both.
# > [...] S4 reparameterizes the structured state matrices $\bm{A}$ from HiPPO by decomposing them as the sum of a low-rank and normal term
# > [...] ultimately reducing to a well-studied and theoretically stable Cauchy kernel.
# > This results in $\widetilde{O}(N+L)$ computation and $O(N+L)$ memory usage, which is essentially tight for sequence models.

# In other words, all of S4's complicated algorithm was to reduce the DPLR SSM kernel to a [Cauchy matrix](https://en.wikipedia.org/wiki/Cauchy_matrix) multiplication which is well-studied and fast.
# In practice, an optimized naive algorithm with $O(NL)$ computation and $O(N+L)$ space is efficient enough.
# This space reduction required a custom kernel in the original PyTorch version of S4.

# **DSS.**
# Although S4 outlined the diagonal case, it focused on the DPLR case for theoretical reasons we'll expand on in Part III.
# DSS found that *truncating S4's matrix to be diagonal* was still empirically effective, and introduced a simple method to take advantage of diagonal SSMs.
# Beyond the choice of diagonal vs DPLR, its parameterization differs from S4's in several ways.
# Most notably, it introduces a **complex softmax** which is specialized to the ZOH discretization and normalizes over the sequence length. These differences were subsequently ablated by S4D which found slight improvements with S4's original design choices.
# 
# <!--
# This was introduced to potentially stabilize the case when $\boldsymbol{A}$ can have positive eigenvalues, but has some disadvantages including being somewhat more complicated and less efficient, and calibrated only to a particular sequence length.
# -->

# **S4D.**
# Presented above, S4D simplified DSS by fleshing out the outline for diagonal kernels based on Vandermonde products, and also theoretically explained the effectiveness of DSS's initialization.
# It found this combination of diagonal initialization together with S4's parameterization to have the best of all worlds: extremely simple to implement, theoretically principled, and empirically effective.


# ***Computational Complexities.***
# Notice that the S4D kernel computation is very similar to the original S4 algorithm in that they both ultimately reduce to a structured matrix vector product (Vandermonde or Cauchy), which actually have the same asymptotic efficiencies.
# In fact, this is no surprise - Vandermonde matrices and Cauchy matrices are very closely related, and have essentially identical computational complexities because they can be easily [transformed to one another](https://arxiv.org/abs/1311.3729).
# It's neat that generalizing the diagonal case to diagonal plus low-rank simply reduces to a slightly different, but computationally equivalent, linear algebra primitive!

# Note that these primitives can be implemented in many ways, which has been the source of some confusion about their efficiencies (is diagonal faster than DPLR?) and implementations (does DPLR require a custom CUDA kernel?).
# In summary, the DPLR kernel (i.e. Cauchy) and all versions of diagonal kernels (i.e. Vandermonde) actually have the *exact same computational complexities* as well as "implementation complexity", because the computational core in all cases is a similar structured matrix product. This can be computed in:
#
# * $O(NL)$ time and $O(NL)$ space, by naively materializing the matrix
# * $O(NL)$ time and $O(N+L)$ space, which either requires a custom kernel (e.g. in PyTorch) or taking advantage of clever compilers (e.g. JAX with XLA) as in our implementation above
# * $\widetilde{O}(N+L)$ time and $O(N+L)$ space theoretically, from a rich body of literature in scientific computing


# ### The Complete S4D Layer
#
# With the core convolutional kernel $\boldsymbol{\overline{K}}$ in place,
# we're ready to put the full S4D layer together!


class S4DLayer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    # The full training script has optimizer hooks that lower the LR on special params
    lr = {
        "A_re": 0.1,
        "A_im": 0.1,
        "B_re": 0.1,
        "B_im": 0.1,
        "log_step": 0.1,
    }

    def setup(self):
        # Learned Parameters
        init_A_re, init_A_im, _, _ = hippo_initializer(self.N)
        self.A_re = self.param("A_re", init_A_re, (self.N,))
        self.A_im = self.param("A_im", init_A_im, (self.N,))
        self.A = np.clip(self.A_re, None, -1e-4) + 1j * self.A_im
        self.B_re = self.param("B_re", nn.initializers.ones, (self.N,))
        self.B_im = self.param("B_im", nn.initializers.zeros, (self.N,))
        self.B = self.B_re + 1j * self.B_im
        self.C = self.param("C", normal(stddev=0.5**0.5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(self.param("log_step", log_step_initializer(), (1,)))
        if not self.decode:
            self.K = s4d_kernel_zoh(self.C, self.A, self.l_max, self.step)
        else:
            # FLAX code to ensure that we only compute discrete once during decoding.
            def init_discrete():
                return s4d_ssm(self.C, self.A, self.l_max, self.step)

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
            return causal_convolution(u, self.K) + self.D * u
        else:
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


S4DLayer = cloneLayer(S4DLayer)


# The core of the S4D layer is the same as the traditional SSM layer defined in the first part of the post. We define our SSM parameters $(\bm{A}, \bm{B}, \bm{C}, \bm{D})$ and then call the kernel code written above as a convolution during training.
# Finally, during discrete decoding, we use the initial recurrence computed above.
# Note that much of the above code is boilerplate for initialization and handling the recurrence case, and the core forward pass (kernel construction and convolution) really only requires < 10 LoC.

# ... and that's all folks! S4D is dramatically more easy to understand and compact than S4, with an extremely structured state space that reduces to a single linear algebra primitive. Together with the new theoretical insights in the next section, we can build a model that is almost as expressive and performant as S4.


# ## Part IIIa. The Central Challenge: Initialization

# The final piece in the above code left unexplained so far is also the most important: how should we initialize the SSM parameters, in particular the diagonal matrix $\boldsymbol{A}$?

# In order to understand this key breakthrough that made diagonal SSMs perform well, we have to briefly revisit the motivation and theoretical interpretation of S4.
# This is the only part of this post that requires some mathematical background, but is optional: the entire model is already fully contained in Parts I and II.

# The initialization is given by the line `hippo_initializer`, which is the diagonal part of the DPLR representation of S4's HiPPO matrix.
# For the rest of this post, we give some historical context and intuition for this initialization.


# ### A Brief Refresher on S4 and HiPPO

# Recall that the critical question for state space models is how to parameterize and initialize the state matrix $\boldsymbol{A}$ in a way that can (i) be computed efficiently and (ii) model complex interactions in the data such as long range dependencies.

# Although the diagonal SSM algorithm presented above is very simple and efficient,
# it's actually extremely difficult to find a diagonal $\boldsymbol{A}$ that performs well!

# As a refresher, S4's motivation was to instead use a particular formula for the $\boldsymbol{A}$ matrix called a [HiPPO matrix](https://arxiv.org/abs/2008.07669) that has a mathematical interpretation of memorizing the history of the input $u(t)$.
# This theory is what gives S4 its remarkable performance on long sequence modeling,
# illustrated in this animation from [[How to Train Your HiPPO](https://link)].

# <center>
# <img src="images/hippo_reconstruction_cropped.gif" width="100%"/>
# An illustration of HiPPO for $L=10000, N=64$.
# </center>

# Here, an input signal $u(t)$ (*Black*) is processed by the HiPPO operator $x' = \boldsymbol{A}x + \boldsymbol{B}u$ (*Blue*) for $10000$ steps, maintaining a state $x(t) \in \mathbb{R}^{64}$. At all times, the current state represents a compression of the history of $u(t)$ and can be linearly projected to approximately reconstruct it (*Red*). This approximation is optimal with respect to an exponentially-decaying measure (*Green*).

# The primary challenge that S4 addressed is how to efficiently compute with this matrix $\boldsymbol{A}$.
# The HiPPO matrix has a simple closed-form formula:
# $$
# \boldsymbol{A} =
# -
# \begin{bmatrix}
# 1 & 0 & 0 & 0 \\
# (3 \cdot 1)^{\frac{1}{2}} & 2 & 0 & 0 \\
# (5 \cdot 1)^{\frac{1}{2}} & (5 \cdot 3)^{\frac{1}{2}} & 3 & 0 \\
# (7 \cdot 1)^{\frac{1}{2}} & (7 \cdot 3)^{\frac{1}{2}} & (7 \cdot 5)^{\frac{1}{2}} & 4 \\
# \end{bmatrix}
# $$

# Note that this matrix is not diagonal, but it is diagonalizable (with eigenvalues $-1, -2, -3, \dots$) - so we can hope to apply Lemma 1.
# Alas, S4 showed that this doesn't work because $\boldsymbol{V}$ has exponentially large entries.

# > Unfortunately, the naive application of diagonalization does not work due to numerical issues.
# [...]
# > The ideal scenario is when the matrix $\bm{A}$ is diagonalizable by a perfectly conditioned (i.e., unitary) matrix.
# > By the Spectral Theorem of linear algebra, this is exactly the class of **normal matrices**.
# > However, this class of matrices is restrictive; in particular, it does not contain the HiPPO matrix.

# This discussion highlights the **key limitation of diagonal SSMs**: although expressive in theory (*algebraically*), they are not necessarily expressive in practice (*numerically*).
# To circumvent this, S4 discovered a new way to put a matrix in *almost diagonal* form, while only needing to conjugate by [unitary matrices](https://en.wikipedia.org/wiki/Unitary_matrix) which are perfectly stable.
#
# $$
# \begin{aligned}
# \boldsymbol{A} &= \boldsymbol{A^{(N)}} - \boldsymbol{P}\boldsymbol{P}^\top \\
# &= -
# \begin{bmatrix}
# \frac{1}{2} & -\frac{1}{2}(3 \cdot 1)^{\frac{1}{2}} & -\frac{1}{2}(5 \cdot 1)^{\frac{1}{2}} & -\frac{1}{2}(7 \cdot 1)^{\frac{1}{2}} \\
# \frac{1}{2}(3 \cdot 1)^{\frac{1}{2}} & \frac{1}{2} & -\frac{1}{2}(5 \cdot 3)^{\frac{1}{2}} & -\frac{1}{2}(7 \cdot 3)^{\frac{1}{2}} \\
# \frac{1}{2}(5 \cdot 1)^{\frac{1}{2}} & \frac{1}{2}(5 \cdot 3)^{\frac{1}{2}} & \frac{1}{2} & -\frac{1}{2}(7 \cdot 5)^{\frac{1}{2}} \\
# \frac{1}{2}(7 \cdot 1)^{\frac{1}{2}} & \frac{1}{2}(7 \cdot 3)^{\frac{1}{2}} & \frac{1}{2}(7 \cdot 5)^{\frac{1}{2}} & \frac{1}{2} \\
# \end{bmatrix}
# - \frac{1}{2}
# \begin{bmatrix}
# 1^{\frac{1}{2}} \\
# 3^{\frac{1}{2}} \\
# 5^{\frac{1}{2}} \\
# 7^{\frac{1}{2}} \\
# \end{bmatrix}
# \begin{bmatrix}
# 1^{\frac{1}{2}} \\
# 3^{\frac{1}{2}} \\
# 5^{\frac{1}{2}} \\
# 7^{\frac{1}{2}} \\
# \end{bmatrix}^\top
# \end{aligned}
# $$

# As discussed in Part 1 of the Annotated S4 [Link], the first component $\boldsymbol{A}^{(N)}$ is a [normal matrix](https://en.wikipedia.org/wiki/Normal_matrix) which is unitarily diagonalizable, hence $\boldsymbol{A}$ is unitarily equivalent to a DPLR matrix.
# This led to all of the fancy machinery to compute the DPLR kernel that S4 introduced.

# ### The Diagonal HiPPO Matrix

# Finally, we can describe the key fact that made diagonal SSMs work.
# The core contribution of DSS is showing that simply **masking out the low-rank portion of the HiPPO matrix** results in a diagonal matrix that empirically performs almost as well as S4.
# This is the key "fork in the road" between the original S4 paper, and the follow-up diagonal SSMs which all use this diagonal approximation of the HiPPO matrix.
# More precisely, they initialize with the matrix $\bm{A}^{(D)}$ defined as the diagonalization (eigenvalues) of the normal matrix $\bm{A}^{(N)}$.

# It can be hard to appreciate how surprising and subtle this fact is.
# It's important to note that writing the HiPPO matrix in DPLR form was S4's main contribution, but *this form was purely for computational purposes*.
# In other words, *the diagonal and low-rank portions by themselves should have no mathematical meaning*.
# In fact, other follow-ups that [generalize and explain S4](https://TODO) introduce different variants of S4 that all have a DPLR representation, but where dropping the low-rank term to convert it to a diagonal matrix performs much worse.

# It turns out that this particular matrix is extremely special, and the diagonal HiPPO matrix *does* have a theoretical interpretation. Dropping the low-rank term - leaving only the normal term $\boldsymbol{A}^{(N)}$ - *has the same dynamics as $\boldsymbol{A}$ in the limit as the state size $N \to \infty$*.
# This is a pretty remarkable fact proved in the S4D paper, and honestly still seems like an incredible coincidence.
# In the rest of this post, we'll unpack this fact and try to give more intuition for SSMs.

# ## Part IIIb. An Intuitive Understanding of SSMs

# We'll close out this blog post with some discussion on how to think about SSMs, illustrated through diagonal SSMs. We'll focus on intuition for the following question:
#
# Q: **How should we interpret the convolution kernel of a state space model**?

# ### Case: 1-dimensional State
# Let's start with the case of an SSM with $N=1$. We'll write lowercase $\bm{a}$ and $\bm{b}$ to emphasize that they're scalars. The state $x(t)$ is then a scalar function that satisfies a linear ODE, which is elementary to solve.
# The original SSM state equation
# $$
# \begin{aligned}
# \frac{d}{dt} x(t) &= \bm{a} x(t) + \bm{b} u(t) \\
# \end{aligned}
# $$
# can be multiplied by a simple term (called an [integrating factor](https://en.wikipedia.org/wiki/Integrating_factor)) to produce a simpler ODE,
# $$
# \begin{aligned}
# \frac{d}{dt} e^{-t\bm{a}} x(t) &= -\bm{a} e^{-t\bm{a}} x(t) + e^{-t\bm{a}}x'(t)
# \\&= e^{-t\bm{a}} \bm{b} u(t)
# .
# \end{aligned}
# $$
# This can be explicitly integrated
# $$
# \begin{aligned}
# e^{-t\bm{a}} x(t) &= \int_0^t e^{-s\bm{a}} \bm{b} u(s) \; ds
# \\
# \end{aligned}
# $$
# which yields a closed formula for the state
# $$
# \begin{aligned}
# x(t) &= \int_0^t e^{(t-s)\bm{a}} \bm{b} u(s) \; ds
# \\&= e^{t\bm{a}} \ast u(t)
# \end{aligned}
# $$

# This shows that the state $x(t)$ is just the (causal) convolution of the input $u(t)$ with an exponential decaying kernel $e^{t\bm{a}}$!

# ### Case: Diagonal SSM

# When $\bm{A}$ is diagonal, the equation $x'(t) = \bm{A} x(t) + \bm{B} u(t)$ can simply be broken into $N$ independent scalar SSMs, as illustrated in the splash figure (*Left*).
# Therefore each element of the state, which we can denote $x_n(t)$, convolves the input by the exponentially decaying kernel $e^{t\bm{A}_n}\bm{B}_n$.
# These kernels are illustrated in the splash figure (*Right*), visualized again here:

# <center>
# <img src="images/basis_diag_real.png" width="60%"/>
# </center>

# The $N$ different scalar equations can be vectorized to just say that the state $x(t) \in \mathbb{R}^N$ is
# $$
# x(t) = (e^{t\bm{A}} \bm{B}) \ast u(t)
# $$
# The above figure very concretely plots (the real part of) the 4 functions $e^{t\bm{A}} \bm{B}$ for
# $$
# \bm{A} =
# \begin{bmatrix}
# -\frac{1}{2} + i \pi & 0 & 0 & 0 \\
# 0 & -\frac{1}{2} + i 2\pi & 0 & 0 \\
# 0 & 0 & -\frac{1}{2} + i 3\pi & 0 \\
# 0 & 0 & 0 & -\frac{1}{2} + i 4\pi \\
# \end{bmatrix}
# \qquad \bm{B} =
# \begin{bmatrix}
# 1 \\ 1 \\ 1 \\ 1 \\
# \end{bmatrix}
# $$

# For example, the first basis function (*Blue*) is just $e^{-\frac{1}{2}t} e^{i\pi t}$.
# Notice how the *real* part of $\bm{A}$ controls the decay of these functions (*dotted lines*),
# while the *imaginary* part controls the frequency!

# What about $y(t)$? This is just a linear combination of the state, $y(t) = \bm{C} x(t)$.
# But by linearity of convolution, we can push $\bm{C}$ inside:
# $$
# y(t) = \bm{C} (e^{t\bm{A}} \bm{B} \ast u(t)) = (\bm{C} e^{t\bm{A}} \bm{B}) \ast u(t)
# $$
# So the entire SSM is equivalent to a 1-D convolution where the kernel is just $\bm{C} e^{t\bm{A}} \bm{B}$.
# We interpret this kernel as **a linear combination of the $N$ basis kernels** $e^{t\bm{A}} \bm{B}$.

# ### Case: General SSM

# The same derivation and formula for the convolution kernel actually holds in the case of non-diagonal state matrices $\bm{A}$,
# only now it involves a **matrix exponential** instead of scalar exponentials.
# We still interpret it the same way: $e^{t\bm{A}} \bm{B}$ is a vector of $N$ different **basis kernels**, and the overall convolution kernel
# $\bm{C} e^{t\bm{A}} \bm{B}$ is a linear combination of these basis functions.

# ### Case: HiPPO and Diagonal-HiPPO

# Equipped with this understanding, we can now try to understand the HiPPO matrix better.
# Although we've been focusing on $\bm{A}$, which is the more important matrix, HiPPO actually provides exact formulas for
# $\bm{A}$ *and* $\bm{B}$.
# So with the above interpretation, *HiPPO provides a specific set of basis functions*, and the parameter $\bm{C}$ then learns a weighted combination of these to use as the final convolution kernel.
# For the particular $(\bm{A}, \bm{B})$ that S4 uses, each basis function actually has a closed-form formula as [exponentially-warped Legendre polynomials](https://TODO) $L_n(e^{-t})$.
# Intuitively, S4's state $x(t)$ convolves the input by each of these very smooth, infinitely-long kernels, which gives rise to its long-range modeling abilities.

# <center>
# <img src="images/basis_legs_clean.png" width="60%"/>
# </center>

# Finally, what about the basis for S4D using this diagonal approximation to the HiPPO matrix?
# Let's plot $e^{t \bm{A}^{(N)}}\frac{\bm{B}}{2}$, in other words the basis kernels for the SSM $(\bm{A}^{(N)}, \bm{B}/2)$.
# Here it is for $N=256$ (*Left*) and $N=1024$ (*Right*):

# <center>
# <img src="images/basis_legsd_256_bilinear.png" width="48%"/>
# <img src="images/basis_legsd_1024_bilinear.png" width="48%"/>
# </center>

# We can see that this matrix $\boldsymbol{A}^{(N)}$ (a dense, diagonalizable matrix) *generates noisy approximations to the same kernels as $\bm{A}$* (a triangular, hard-to-diagonalize matrix) that are *exactly equal* as $N\to\infty$.
# This is what we meant by saying the diagonal-HiPPO matrix is a perfect approximation to the original HiPPO matrix, which really seems like a remarkable mathematical coincidence!

# Just to drive home the point, let's show some other bases. Once again, the normal-HiPPO matrix is $\bm{A}^{(N)} = \bm{A} + \bm{P}\bm{P}^\top$ for a matrix $\bm{P}$ with entries of order $N^{\frac{1}{2}}$.
# What happens if we replace $\bm{P}$ with a random vector? The basis quickly becomes unstable even for very small magnitudes of $\bm{P}$!

# <center>
# <img src="images/basis_legs_std_03.png" width="32%"/>
# <img src="images/basis_legs_std_04.png" width="32%"/>
# <img src="images/basis_legs_std_05.png" width="32%"/>
# Random low-rank perturbation $\bm{P}$, i.i.d. Gaussian with standard deviations $\sigma = 0.3, 0.4, 0.5$.
# </center>


# Finally, what happens with other DPLR matrices? A follow-up theoretical paper to S4 and HiPPO derived other variants, for example a new SSM $(\bm{A}, \bm{B})$ that produces **truncated Fourier** basis functions (*Left*). This is particularly useful as a way to generalize standard CNNs, since the basis kernels are local.
# This matrix $\bm{A}$ can also be written in DPLR form, so it can be computed efficiently with S4 (a variant called S4-FouT).
# But the same trick of dropping the low-rank term produces basis functions that are qualitatively quite different - oscillating infinitely instead of being truncated (*Right*) - and performs quite poorly empirically!

# <center>
# <img src="images/basis_fout_1024.png" width="48%"/>
# <img src="images/basis_fout_norank.png" width="48%"/>
# </center>

# It's worth repeating: the particular HiPPO variant that S4 uses, and the fact that a particular low-rank correct makes it normal/diagonal while preserving the same basis, is *really* special!

# ### Other Diagonal Initializations

# Inspired by the diagonal-HiPPO matrix, S4D proposes several new initializations with simpler formulas where the real part is fixed to $-\frac{1}{2}$ and the imaginary part follows a polynomial law.
# In fact, the example in the previous section with $\boldsymbol{A}_n = -\frac{1}{2} + i \pi n$ is the simplest variant, called **S4D-Lin** because the imaginary part scales linearly in $n$.

# However, for now, it seems like the full HiPPO matrix is core to S4's long-range modeling abilities, and the diagonal-HiPPO initialization also seems empirically best among diagonal SSMs.
# The following table shows results for several variants of S4 and S4D with various $(\boldsymbol{A}, \boldsymbol{B})$ initializations, which all define difference bases functions and have different strengths.

# <center>
# <img src="images/lra.png" width=95%"/>
#
# **Long Range Arena (LRA)**. (*Top*) S4 variants with DPLR $\bm{A}$; LegS is equivalent to the original S4.
# (*Middle*) S4D variants; LegS refers to diagonal-HiPPO $\bm{A}^{(D)}$.
# (*Bottom*) Previous results, including the original S4 which had a different architecture and hyperparameters.
# </center>

# We see that all of them perform very well in general, and the very simple S4D-Lin initialization is even best on several of the 5 main tasks.
# However, the original S4 initialization and its diagonal approximation are so far the only ones that can solve Path-X.

# **Open challenge**: are there alternative SSMs not based on HiPPO that can get to 90% on Path-X?

# The S4D paper performed a careful empirical studying ablating the parameterization and initializations of SSM variants, and found that controlling for all other hyperparameters, S4's full DPLR representation is often slightly better than S4D's diagonal variant, especially for harder tasks. However, diagonal SSMs can certainly provide a lot more bang-for-your-buck in terms of complexity to payoff, and we highly recommend starting here for understanding SSMs and exploring them for applications!

# ## Conclusion

# The introduction of new **diagonal state space models** show the potential of SSMs as sequence models that can be incredibly powerful, yet quite simple to understand and implement.
# Many directions are open for exploration, from fundamental research in understanding and improving these models, to drawing connections to the rich scientific literature on state spaces, to exploring direct applications in domains such as audio, vision, time-series, NLP, and more.
# In writing this post, we hope that fleshing out the details of these models can lower the barrier to understanding S4 and inspire future ideas in this area.
# There's much left to understand here, and we believe that perhaps even simpler and better models will be uncovered!

# [**Final citations / links to resources**]

# [**Acknowledgements**]

