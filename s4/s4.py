# # S4

import jax.numpy as np
import jax
from jax.scipy.signal import convolve
from jax.numpy.linalg import matrix_power as power
from jax.numpy.linalg import inv, eigvals, eig

# Apply at the roots of unity
#
# $$\Omega_M = \{\exp(2 \pi i  \frac{k}{M}) : k \in [M])\}$$

def applyAtRootsOfUnity(fn, M):
    r = np.exp((2j * np.pi / M) * np.arange(M))
    return jax.vmap(fn)(r)


key = jax.random.PRNGKey(0)
def Param(shape):
    return jax.random.uniform(key, shape)


class SSM:
    def __init__(self, A, B, C, discrete=False):
        self.A = A
        self.B = B
        self.C = C
        assert A.shape[0] == C.shape[1]
        assert A.shape[1] == B.shape[0]
        self.N = self.A.shape[0]
        self.discrete = discrete
        


def make_HiPPO(N):
    p = np.sqrt(2 * np.arange(1, N+1) + 1.)
    q = p
    A = p[:, np.newaxis] @ q[np.newaxis, :]
    hippo = -np.tril(A, k=-1) - np.diag(np.arange(1, N+1)+1)
    print(hippo)
    S = hippo + 0.5 * A + 0.5 * np.eye(N)
    # Skew symmetric
    print (S)
    diag = eigvals(S)
    diag = diag - 0.5
    _, v = eig(S)
    return hippo, diag, 0.5 * p, q, v


out, diag, p, q, V = make_HiPPO(4)
out

out2 = V @ np.diag(diag) @ V.conj().T - p[:, np.newaxis] @ q[np.newaxis, :]
out2

def discretizeSSM_bilinear(ssm, step):
    I = np.eye(ssm.N)
    left = inv((I - (step / 2.) * ssm.A))
    Abar = left @ (I + (step / 2.) * ssm.A)
    Bbar = (left * step) @ ssm.B
    Cbar = ssm.C
    return SSM(Abar, Bbar, Cbar, discrete=True)


def K_conv_naive(ssm, L):
    assert ssm.discrete
    return np.array([(ssm.C @ power(ssm.A, l) @ ssm.B).reshape()
                     for l in range(L)])

def K_gen_naive(ssm, L):
    assert ssm.discrete
    coef = K_conv_naive(ssm, L)
    def gen(o):
        terms = [coef[l] * (o**l) for l in range(L)]
        return sum(terms)
    return gen


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

# Apply non-circular convolution in fourier space
def nonCircularConvolution(x, out):
    return convolve(x, out, method='fft')



class S4:
    def __init__(self, L, N, step=1):
        self.L = L
        self.N = N
        self.step = step

        # Factored A


        _, diag, p, q, _ = make_HiPPO(N)
        self.Gamma = diag
        self.p = p
        self.q = q
        self.B = Param((N, 1))
        self.C = Param((1, N))
        self.A = np.diag(self.Gamma) - self.p[:, np.newaxis] * self.q[np.newaxis, :].conj()

        self.ssm = SSM(self.A, self.B, self.C)
        self.discrete = discretizeSSM_bilinear(self.ssm, step)
        
        self.Ct = (np.eye(N) - power(self.discrete.A, self.L)).conj().T @ self.discrete.C.ravel()
        # self.Ct = self.discrete.C @ (np.eye(N) - power(self.discrete.A, self.L))
        
    def K_gen(self):
        return ssmGeneratingFn((self.Ct.conj().ravel(), self.q.conj().ravel()),
                              (self.B.ravel(), self.p.ravel()),
                              self.step, self.Gamma)
    
    def forward(self):
        self.B



# L = 16000
# s = S4(L, 2)
# out2 = s.K_gen()
# out2 = convFromGen(out2, L)
# out2
