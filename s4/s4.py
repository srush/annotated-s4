# # S4

import jax.numpy as np
import jax
import matplotlib.pyplot as plt
from jax.scipy.signal import convolve
from jax.numpy.linalg import matrix_power as power
from jax.numpy.linalg import inv, eigvals, eig
key = jax.random.PRNGKey(0)

x = np.linspace(0, 10, 64)
y = np.sin(x) + jax.random.uniform(key, (64,))

plt.plot(x, y)
__st.pyplot()


# Create a state space model.

class SSM:
    def __init__(self, A, B, C, discrete=False):
        self.A = A
        self.B = B
        self.C = C
        assert A.shape[0] == C.shape[1]
        assert A.shape[1] == B.shape[0]
        self.N = self.A.shape[0]
        self.discrete = discrete

        
# Discretize SSM

def discretizeSSM_bilinear(ssm, step):
    I = np.eye(ssm.N)
    BL = inv((I - (step / 2.) * ssm.A))
    Abar = BL @ (I + (step / 2.) * ssm.A)
    Bbar = (BL * step) @ ssm.B
    Cbar = ssm.C
    return SSM(Abar, Bbar, Cbar, discrete=True)

# Make a conv filter version of SSM

def K_conv_naive(ssm, L):
    assert ssm.discrete
    return np.array([(ssm.C @ power(ssm.A, l) @ ssm.B).reshape()
                     for l in range(L)])

# Create the HiPPO matrix

# Formula

def make_HiPPO(N):
    p = np.sqrt(2 * np.arange(1, N+1) + 1.)
    A = p[:, np.newaxis] @ p[np.newaxis, :]
    return -np.tril(A, k=-1) - np.diag(np.arange(1, N+1)+1)

# Apply non-circular convolution in fourier space
def nonCircularConvolution(x, filt):
    return convolve(x, filt, mode="full", method='fft')[:x.shape[0]-1]

# Create a model
def Param(shape):
    return jax.random.uniform(key, shape)

N1 = 16

A, B, C = make_HiPPO(N1), Param((N1, 1)), Param((1, N1))

def model(ssm, x):
    ssm = discretizeSSM_bilinear(ssm, 1)
    conv = K_conv_naive(ssm, 64)
    return nonCircularConvolution(x, conv)

def loss(params):
    B, C = params
    yhat = model(SSM(A, B, C), y)
    return -np.square((y[1:] - yhat)).mean()

g = jax.value_and_grad(loss)
params = [B, C]
for i in range(10):
    loss, gParams = g(params)
    params = [p + 0.1 * g for p, g in zip(params, gParams)]
    print(loss)


ssm = discretizeSSM_bilinear(SSM(A, params[0], params[1]), 1)
X = ssm.B @ y[0:1]
v = []
for i in range(1, 64):
    X = ssm.A @ X + ssm.B @ y[i:i+1]
    n = ssm.C @ X
    v.append(n)
plt.plot(x[1:64], v)
plt.plot(x[1:64], y[1:])
__st.pyplot()



# Apply at the roots of unity
#
# $$\Omega_M = \{\exp(2 \pi i  \frac{k}{M}) : k \in [M])\}$$

def K_gen_naive(ssm, L):
    assert ssm.discrete
    coef = K_conv_naive(ssm, L)
    def gen(o):
        terms = [coef[l] * (o**l) for l in range(L)]
        return sum(terms)
    return gen

def applyAtRootsOfUnity(fn, M):
    r = np.exp((2j * np.pi / M) * np.arange(M))
    return jax.vmap(fn)(r)


def K_gen_inverse(ssm, L):
    assert ssm.discrete
    I = np.eye(ssm.A.shape[0])
    A_L = power(ssm.A, L)
    def gen(o):
        return (ssm.C @ (I - A_L) @ inv(I - ssm.A * o) @ ssm.B).reshape()
    return gen


def convFromGen(gen, L):
    order = np.array([i if i == 0 else L - i for i in range(L)])
    K_hat = applyAtRootsOfUnity(gen, L)
    out = np.fft.ifft(K_hat, L).reshape(L)
    return out[order]


# Compute a Cauchy dot product $$m v$$ where
#
# $$m_{j} = \frac{1}{\omega - \lambda_j}$$ 

def cauchy_dot(v, omega, lambd):
    return (v / (omega - lambd)).sum()

def ssmGeneratingFn(aterm, bterm, step, diag):
    assert aterm[0].shape == aterm[1].shape
    assert aterm[0].shape == bterm[0].shape
    assert aterm[0].shape == bterm[1].shape
        
    def gen(o):
        f = (2. / step) * ((1. - o) / (1. + o))
        k00 = cauchy_dot(aterm[0] * bterm[0], f, diag)
        k01 = cauchy_dot(aterm[0] * bterm[1], f, diag)
        k10 = cauchy_dot(aterm[1] * bterm[0], f, diag)
        k11 = cauchy_dot(aterm[1] * bterm[1], f, diag)
        return (2. / (1. + o)) * (k00 - k01 * (1. / (1. + k11)) * k10)
    return gen


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

class S4:
    def __init__(self, L, N, step=1):
        self.L = L
        self.N = N
        self.step = step

        # Factored A
        _, diag, p, q, _ = make_DPLR_HiPPO(N)
        self.Gamma = diag
        self.p = p
        self.q = q
        self.A = np.diag(self.Gamma) - self.p[:, np.newaxis] * self.q[np.newaxis, :].conj()
        self.step = step
        self.N = N
        self.set_params([Param((N, 1)), Param((1, N))])

        
    def K_gen(self):
        return ssmGeneratingFn((self.Ct.conj().ravel(), self.q.conj().ravel()),
                               (self.B.ravel(), self.p.ravel()),
                               self.step, self.Gamma)

    def get_params(self):
        return [self.B, self.C]

    def set_params(self, params):
        self.B, self.C = params
        self.ssm = SSM(self.A, self.B, self.C)
        self.discrete = discretizeSSM_bilinear(self.ssm, self.step)
        self.Ct = (np.eye(self.N) - power(self.discrete.A, self.L)).conj().T @ self.discrete.C.ravel()

M = S4(64, N1)
def model(params, x):
    M.set_params(params)
    out = M.K_gen()
    conv = convFromGen(out, M.L)
    return nonCircularConvolution(x, conv)

def loss(params):
    yhat = model(params, y)
    return -np.square((y[1:] - yhat.real)).mean()

g = jax.value_and_grad(loss)

for i in range(10):
    params = M.get_params()
    loss, gParams = g(params)
    params = [p + 0.1 * g for p, g in zip(params, gParams)]
    M.set_params(params)
    print(loss)

ssm = M.discrete
X = ssm.B @ y[0:1]
v = []
for i in range(1, 64):
    X = ssm.A @ X + ssm.B @ y[i:i+1]
    n = ssm.C @ X
    v.append(n)
plt.plot(x[1:64], v)
plt.plot(x[1:64], y[1:])
__st.pyplot()


# L = 16000
# s = S4(L, 2)
# out2 = s.K_gen()
# out2 = convFromGen(out2, L)
# out2
