from functools import partial
import jax
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from flax.training import checkpoints
from .data import Datasets
from .s4 import S4LayerInit
from .train import BatchSeqModel


if __name__ == "__main__":

    model = S4LayerInit(N=64)
    model = partial(
        BatchSeqModel, layer=model, d_output=256, d_model=128, n_layers=4, l_max=784
    )
    model = model(training=False)
    rng = jax.random.PRNGKey(0)
    state = checkpoints.restore_checkpoint("checkpoints/mnist/s4-d_model=128/", None)
    _, testloader, _, _, _ = Datasets["mnist"](bsz=1)
    i = iter(testloader)
    next(i)
    next(i)
    next(i)
    # next(i)
    cur = np.array(next(i)[0].numpy())
    # cur = np.zeros((1, 784, 1))

    def loop(i, cur):
        cur, rng = cur
        r, rng = jax.random.split(rng)
        out = model.apply({"params": state["params"]}, cur)
        # sample with temperature
        # p = (np.exp(out[0, i] * 1.1) * np.arange(256)).sum(-1).round().astype(int)
        p = jax.random.categorical(rng, out[0, i])
        # * 1.1) * np.arange(256)).sum(-1).round().astype(int)
        cur = jax.ops.index_update(cur, (0, i + 1, 0), p)
        return cur, rng

    # start  pixel
    start = 500
    out = jax.lax.fori_loop(start, 783, jax.jit(loop), (cur, rng))[0]
    out = out.reshape(28, 28)
    print(out)
    final = onp.zeros((28, 28, 3))
    final[:, :, 0] = out

    final[:, :, 1] = cur.reshape(28, 28)
    final.reshape(28 * 28, 3)[:start, 2] = cur.reshape(28 * 28)[:start]
    plt.imshow(final / 256.0)
    plt.savefig("im.png")
