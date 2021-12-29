from functools import partial
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from flax.training import checkpoints, train_state
from .data import Datasets
from .s4 import NaiveSSMInit, S4LayerInit
from .train import BatchSeqModel


if __name__ == "__main__":

    model = S4LayerInit(N=64)
    model = partial(
        BatchSeqModel, layer=model, d_output=256, d_model=64, n_layers=4, l_max=784
    )
    model = model(training=False)
    rng = jax.random.PRNGKey(0)
    state = checkpoints.restore_checkpoint("mnist.old", None)
    _, testloader, _, _ = Datasets["mnist"](bsz=1)
    i = iter(testloader)
    next(i)
    next(i)
    next(i)
    next(i)
    cur = np.array(next(i)[0].numpy())
    # cur = np.zeros((1, 784, 1))

    def loop(i, cur):
        cur, rng = cur
        r, rng = jax.random.split(rng)
        out = model.apply({"params": state["params"]}, cur)
        # sample with temperature
        p = (np.exp(out[0, i + 1]) * np.arange(256)).sum(-1).round()
        cur = jax.ops.index_update(cur, (0, i + 1, 0), p)
        return cur, rng

    # start at the 300th pixel
    out = jax.lax.fori_loop(300, 783, jax.jit(loop), (cur, rng))[0]
    out = out.reshape(28, 28)
    print(out)
    plt.imshow(out)
    plt.savefig("im.png")
