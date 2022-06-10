from functools import partial
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import seaborn
from flax.training import checkpoints
from .data import Datasets
from .s4 import S4Layer
from .s4d import S4DLayer
from .train import BatchStackedModel


seaborn.set_context("paper")

if __name__ == "__main__":
    layer_args = {}
    layer_args["N"] = 64
    layer_args["l_max"] = 784

    model = S4DLayer
    model = partial(
        BatchStackedModel,
        layer_cls=model,
        layer=layer_args,
        d_output=256,
        d_model=512,
        n_layers=6,
        prenorm=True,
        classification=False,
        decode=True
    )

    rng = jax.random.PRNGKey(0)
    state = checkpoints.restore_checkpoint(
        "checkpoints/mnist/{'d_model': 512, 'n_layers': 6, 'dropout': 0.0, 'prenorm': True, 'layer': {'N': 64, 'l_max': 784}}-d_model=512-lr=0.005-bsz=32", None
    )
    
    _, testloader, _, _, _ = Datasets["mnist"](bsz=1)
    # print(validate(state["params"], model, testloader, classification=False))
    model = model(training=False)
    it = iter(testloader)
    for j, im in enumerate(it):
        if j < 10:
            continue
        print(j)
        image = np.array(im[0].numpy())
        # cur = np.zeros((1, 783, 1))
        cur = image

        def loop(i, cur):
            cur, rng = cur
            r, rng = jax.random.split(rng)
            out = model.apply({"params": state["params"]}, cur[:, :-1])
            # print(i, (np.exp(out[0, i]) * np.arange(256)).sum(-1))
            # print(i, (np.exp(out[0, i])[cur[0, i + 1, 0]]))
            # print(i, cur[0, i], cur[0, i + 1])
            # print("b", out[0, i])
            # print("c", np.exp(out[0, i]))

            # print(i, np.exp(out[0, i]) * np.arange(256).astype(float))
            # p = (np.exp(out[0, i]) * np.arange(256).astype(float)).sum(-1).round().astype(int)

            # sample with temperature

            p = jax.random.categorical(rng, out[0, i] * 1.02)
            # * 1.1) * np.arange(256)).sum(-1).round().astype(int)
            cur = jax.ops.index_update(cur, (0, i + 1, 0), p)
            return cur, rng

        # start  pixel
        start = 300
        # cur = (cur, rng)
        # for j in range(start, start + 10):
        #     cur = loop(j, cur)
        for i in range(start + 3, 784):
            cur = jax.ops.index_update(cur, (0, i, 0), 0)
        print("start")
        out = jax.lax.fori_loop(start, 783, jax.jit(loop), (cur, rng))[0]
        print("end")
        out = out.reshape(28, 28)
        # print(out)
        # print(image.reshape(28, 28))
        final = onp.zeros((28, 28, 3))
        final[:, :, 0] = out
        final.reshape(28 * 28, 3)[:start, 1] = image.reshape(28 * 28)[:start]
        final.reshape(28 * 28, 3)[:start, 2] = image.reshape(28 * 28)[:start]

        final2 = onp.zeros((28, 28, 3))
        final2[:, :, 1] = image.reshape(28, 28)
        final2.reshape(28 * 28, 3)[:start, 0] = image.reshape(28 * 28)[:start]
        final2.reshape(28 * 28, 3)[:start, 2] = image.reshape(28 * 28)[:start]

        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.set_title("Sampled")
        ax1.imshow(final / 256.0)
        ax2.set_title("True")
        ax1.axis("off")
        ax2.axis("off")
        ax2.imshow(final2 / 256.0)
        fig.savefig("im%d.png" % (j))

        if j > 100:
            break
