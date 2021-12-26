from flax.training import train_state, checkpoints
from .train import BatchSeqModel
from .data import Datasets
from .s4 import S4LayerInit, NaiveSSMInit
from functools import partial
import jax.numpy as np
import jax
import matplotlib.pyplot as plt

if __name__ == "__main__":
   model = S4LayerInit(N=64)
   model = partial(
        BatchSeqModel, layer=model, d_output=256, d_model=64, n_layers=4, l_max=784
   )
   model = model(training=False)
   rng = jax.random.PRNGKey(0)
   state = checkpoints.restore_checkpoint("mnist", None)
   cur = np.zeros((1, 784, 1)) 

   def loop(i, cur):
      cur, rng = cur
      r, rng = jax.random.split(rng)
      out = model.apply({"params": state["params"]}, cur)
      p = jax.random.categorical(r, out[0, i+1] / 1.5)
      cur = jax.ops.index_update(cur, (0, i+1, 0), p)
      return cur, rng
   out = jax.lax.fori_loop(0, 783, jax.jit(loop), (cur, rng))[0]
   out = out.reshape(28, 28)
   print(out)
   plt.imshow(out)
   plt.savefig("im.png")
