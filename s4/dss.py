import s4.s4 as s4

from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn



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

class DSSLayer(nn.Module):
    Lambda: np.DeviceArray
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # Learned Parameters
        self.W = self.param(
            "W", s4.lecun_normal(dtype=np.complex64), (1, self.N)
        )
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(self.param("log_step", s4.log_step_initializer(), (1,)))
        self.K = dss_kernel(self.W, self.Lambda, self.l_max, self.step)

    def __call__(self, u):
        return s4.non_circular_convolution(u, self.K) + self.D * u


DSSLayer = s4.cloneLayer(DSSLayer)
    
def DSSLayerInit(N):
    _, Lambda, _, _, _ = s4.make_NPLR_HiPPO(2*N)
    Lambda = Lambda[np.nonzero(Lambda.imag > 0, size=N)]
    return partial(DSSLayer, N=N, Lambda=Lambda)



# def make_normal(N):
#     def v(i, j):
#         if i > j:
#             return np.sqrt(2 * i + 1) * np.sqrt(2 * j + 1)
#         elif i == j:
#             return -1
#         else:
#             return -np.sqrt(2 * i+1) * np.sqrt(2 * j + 1)

#     # Do it slow so we don't mess it up :)
#     mat = [[v(n, k) for k in range(1, N + 1)] for n in range(1, N + 1)]
#     return np.array(mat) / 2.0


# def test_normal():
#     n = make_normal(4)
#     Lambda, V = jax.jit(s4.eig, backend="cpu")(n)
#     _, L2, _, _, _ = s4.make_NPLR_HiPPO(4)
#     print(Lambda)
#     print(L2)
#     assert np.isclose(Lambda, L2).all()

