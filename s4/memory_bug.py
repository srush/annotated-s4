from functools import partial
import jax
import jax.numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from jax.nn.initializers import normal
from tqdm import tqdm


rng = jax.random.PRNGKey(1)

@jax.jit
def cauchy(omega, lambd):
    """ signature: (l), (n) -> (l) """
    cauchy_dot = lambda _omega: (1. / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)

class TestLayer(nn.Module):
    N: int
    L: int

    def setup(self):
        # self.x = self.param("x", normal(dtype=np.complex64), (self.N,))
        # self.y = self.param("y", normal(dtype=np.complex64), (self.N,))
        self.x = self.param("x", normal(), (self.N, 2))
        self.x = self.x[..., 0] + 1j * self.x[..., 1]
        self.y = self.param("y", normal(), (self.N, 2))
        self.y = self.y[..., 0] + 1j * self.y[..., 1]

        self.z = cauchy(np.arange(self.L), self.x*self.y).real

    def __call__(self, u):
        return u + self.z

# Broadcast over (C) channels
# The cauchy() call maps params x, y of shape (N, C) to z of shape (L, C)
TestLayer = nn.vmap(
    TestLayer,
    in_axes=1,
    out_axes=1,
    variable_axes={"params": 1},
    split_rngs={"params": True},
)

# Broadcast over (B) batch
TestLayer = nn.vmap(
    TestLayer,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None},
    split_rngs={"params": False},
)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", type=int, default=16) # batch size
    parser.add_argument("-C", type=int, default=256) # broadcast channelsh
    parser.add_argument("-N", type=int, default=256) # state size N
    parser.add_argument("-L", type=int, default=1024) # seq len L
    args = parser.parse_args()

    in_shape = (args.B, args.L, args.C) # Input shape

    model = TestLayer(N=args.N, L=args.L)
    params = model.init({"params": rng}, np.ones(in_shape))

    # Create optimizer
    tx = optax.adam(learning_rate=1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @partial(jax.jit, static_argnums=(2,))
    def train_step(state, inputs, model):
        def loss_fn(params):
            outputs = model.apply(params, inputs)
            loss = np.mean(outputs)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # Loop over steps
    for step in tqdm(range(1000000)):
        inputs = jax.random.normal(rng, in_shape)
        state, _ = train_step(state, inputs, model)
