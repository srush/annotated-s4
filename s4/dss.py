import s4.s4 as s4

from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal
from jax.numpy.linalg import eig

rng = jax.random.PRNGKey(1)


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
    return (Abar, Bbar, Cbar)


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
                return dss_ssm(self.self.Lambda, self.l_max, self.step)
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
    "Maybe this a general test?"
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
