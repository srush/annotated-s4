# # S4

# ## Efficiently Modeling Long Sequences with Structured State Spaces
# https://arxiv.org/abs/2111.00396


import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax.scipy.signal import convolve
from jax.numpy.linalg import matrix_power as power
from jax.numpy.linalg import inv, eigvals, eig
import torchvision
import torch
import optax
from flax import linen as nn
import flax
from flax.training import train_state
import torchvision.transforms as transforms

# Problem Definition

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(1, 784).t())
])

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
valset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=10)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=128, shuffle=False, num_workers=10)


class MNistModel(nn.Module):
    layer : nn.Module
    d_output : int =256
    d_model : int =256
    n_layers : int = 4 
    dropout : float =0.2
    
    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        layers = []
        norms = []
        dropouts = []
        for _ in range(self.n_layers):
            layers.append(self.layer)
            norms.append(nn.LayerNorm())
            dropouts.append(nn.Dropout(self.dropout, deterministic=False))
        self.layers = layers
        self.norms = norms
        self.dropouts = dropouts
        # Linear decoder
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
            return x
        return jax.vmap(run)(x)

# Make this an RNN or CNN for sanity?
    
class Layer(nn.Module): 
    def setup(self):
        pass
    def __call__(self, x):
        return x

def cross_entropy_loss(*, logits, labels):
    one_hot_labels = jax.nn.one_hot(labels[..., 0], num_classes=256)
    return -np.mean(np.sum(one_hot_labels * logits, axis=-1))

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
rng, dropout_rng = jax.random.split(rng)


# Init
def create_train_state(model, init_rng, dropout_rng, learning_rate, momentum):
  
    params = model.init({"params" : init_rng, "dropout" : dropout_rng},
                        np.ones((128, 784, 1)))['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

# loop
def train_step(state, batch, model):    
    def loss_fn(params):
        logits = model.apply({'params': params}, batch[:, :-1], rngs={"dropout" : dropout_rng})
        loss = cross_entropy_loss(logits=logits[:, :batch.shape[1]], labels=batch[:, 1:])
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

def train_epoch(state, model):
    batch_metrics = []
    for batch_idx, (inputs, targets) in enumerate(valloader):
        inputs = np.array(inputs.numpy())
        state = train_step(state, inputs, model)

# model = MNistModel(Layer)
# train_epoch(create_train_state(model, init_rng, dropout_rng, 0.1, 0.01), model)




# # Part 1: The Model

# The state space model 

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
    BL = inv((I - (step / 2.) * A))
    Abar = BL @ (I + (step / 2.) * A)
    Bbar = (BL * step) @ B
    return (Abar, Bbar, C)


# This allows us to run the model as a recurrent network.

# $x_{t+1} = \mathbf{\bar{A}} x_t + \mathbf{\bar{B}} u_t$

# $y_t = \mathbf{\bar{C}} x_t$

# RNN 

def iterative_SSM(A, B, C, y):
    def f(X, y):
        X = A @ X + (B * y).ravel()
        return (X, C @ X)
    return jax.lax.scan(f, np.zeros(B.shape[0]), y)[1]


# ## Convolution

# Because each step of the SSM is linear we can
# also compute $y_t$ without a recurrence.

# CNN

# Make a conv filter version of SSM

# $K = (\bar{C} \bar{B}, \bar{C} \bar{A}^1 \bar{B}, \ldots, \bar{C} \bar{A}^{L-1} \bar{B})$

# We call $K \in R^L$ the convolutional filter representation of the discrete SSM.

def K_conv(A, B, C, L):
    return np.array([(C @ power(A, l) @ B).reshape()
                     for l in range(L)])


# We can then compute $y$ by convolution with this $K$.

# This convolution is big though! 

def nonCircularConvolution(x, filt):
    return convolve(x, filt, mode="full", method='fft')[:x.shape[0]]


# ## HiPPO matrices

# For this model to work, initialization is really important.

def make_HiPPO(N):
    p = np.sqrt(2 * np.arange(1, N+1) + 1.)
    A = p[:, np.newaxis] @ p[np.newaxis, :]
    return -np.tril(A, k=-1) - np.diag(np.arange(1, N+1)+1)

# ## Running it 

# Create a model
N1 = 16
L1 = 64
LR = 0.1
# A, B, C = make_HiPPO(N1), Param((N1, 1)), Param((1, N1))
# params = [B, C]

# def model(params, y):
#     ssm = discretize_SSM(A, params[0], params[1])
#     # Turn to convolution
#     K = K_conv(*ssm, L1)
#     # Run as convolution
#     return nonCircularConvolution(y, K)


class S4Layer(nn.Module):
    H : int = 50
    l_max : int = 16
    dropout : float = 0.2
    N : int = 50
    
    def setup(self):
        self.A = make_HiPPO(self.N).reshape(1, self.N, self.N)
        self.B = self.param('B', nn.initializers.zeros, (self.H, self.N, 1))
        self.C = self.param('C', nn.initializers.zeros, (self.H, 1, self.N))
    
    def __call__(self, y):
        def create_ssms(A, B, C):
            ssm = discretize_SSM(A, B, C)
            return K_conv(*ssm, self.l_max)
        K = jax.vmap(lambda B, C: create_ssms(self.A, B, C))(self.B, self.C)

        # Run as convolution
        return jax.vmap(nonCircularConvolution)(y, K)
    # return x, None




def run_train():

    # Run the ssm
    ssm = discretize_SSM(A, params[0], params[1])
    v = iterative_SSM(*ssm, y)

    plt.plot(x[1:65], v[1:])
    plt.plot(x[1:65], y[1:])
    __st.pyplot()

    
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
    return lambda z: np.sum(K * (z**np.arange(L)))


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
        f = (2. / step) * ((1. - o) / (1. + o))
        k00 = cauchy_dot(aterm[0] * bterm[0], f, Gamma)
        k01 = cauchy_dot(aterm[0] * bterm[1], f, Gamma)
        k10 = cauchy_dot(aterm[1] * bterm[0], f, Gamma)
        k11 = cauchy_dot(aterm[1] * bterm[1], f, Gamma)
        return (2. / (1. + o)) * (k00 - k01 * (1. / (1. + k11)) * k10)
    return gen


# ## Turning HiPPO to DPLR

def make_DPLR_HiPPO(N):
    p = np.sqrt(2 * np.arange(1, N+1) + 1.)
    q = p
    A = p[:, np.newaxis] @ q[np.newaxis, :]
    hippo = -np.tril(A, k=-1) - np.diag(np.arange(1, N+1)+1)
    S = hippo + 0.5 * A + 0.5 * np.eye(N)
    # Skew symmetric
    diag = eigvals(S)
    diag = diag - 0.5
    _, v = eig(S)
    return hippo, diag, 0.5 * p, q, v


# ## The Model
class S4(nn.Module):
    H : int = 50
    L : int = 16
    N : int = 50
    step : int = 1
   
    def setup(self):
        self.A, self.Gamma, self.p, self.q, _ = make_DPLR_HiPPO(self.N)
        self.B = self.param('B', nn.initializers.zeros, (self.H, self.N, 1))
        self.C = self.param('C', nn.initializers.zeros, (self.H, 1, self.N))
       
        Abar, _, Cbar = discretize_SSM(self.A, self.B, self.C, self.step)
        self.Ct = jax.vmap(
            lambda Cbar: (np.eye(self.N) - power(Abar, self.L)).conj().T @ Cbar.ravel())(Cbar)
   
    def __call__(self, y):
        def create_ssms(B, Ct):
            K_gen = K_gen_DPLR(self.Gamma, self.p, self.q, B, Ct, self.step)
            return convFromGen(K_gen, self.L)
        K = jax.vmap(create_ssms)(self.B, self.Ct)
        return jax.vmap(nonCircularConvolution)(y, K)


model = MNistModel(layer=S4(H=256, L=64))
train_epoch(create_train_state(model, init_rng, dropout_rng, 0.1, 0.01), model)

    

# # Part 3: Pathing.


# ## Path-X


# 

# L = 16000
# s = S4(L, 2)
# out2 = s.K_gen()
# out2 = convFromGen(out2, L)
# out2
