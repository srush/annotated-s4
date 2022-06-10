from .s4 import S4Layer, BatchStackedModel, sample_checkpoint, sample_mnist_prefix
import matplotlib.pyplot as plt
from .s4d import S4DLayer
import jax


rng = jax.random.PRNGKey(1)
def DefaultMNIST(l):
    layer_args = {}
    layer_args["N"] = 64
    layer_args["l_max"] = 784

    # TODO -> Read this from file information.?
    # model = S4DLayer
    # model = BatchStackedModel(
    #     layer_cls=model,
    #     layer=layer_args,
    #     d_output=256,
    #     d_model=512,
    #     n_layers=6,
    #     prenorm=True,
    #     classification=False,
    #     decode=True,
    #     training=False
    # )
    model = S4Layer
    model = BatchStackedModel(
        layer_cls=model,
        layer=layer_args,
        d_output=256,
        d_model=512,
        n_layers=6,
        prenorm=False,
        classification=False,
        decode=True,
        training=False
    )
    return model


MNIST_LEN = 784
default_train_path = "best_16"
# default_train_path = "checkpoints/mnist/{'d_model': 512, 'n_layers': 6, 'dropout': 0.0, 'prenorm': True, 'layer': {'N': 64, 'l_max': 784}}-d_model=512-lr=0.004-bsz=32/"
# default_train_path = "/home/srush/best_13"
out = sample_checkpoint(default_train_path, DefaultMNIST(MNIST_LEN), MNIST_LEN, rng)
plt.imshow(out.reshape(28, 28))
plt.savefig("sample.png")

out = sample_mnist_prefix(default_train_path, DefaultMNIST(MNIST_LEN), MNIST_LEN, rng)

