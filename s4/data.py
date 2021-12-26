import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import numpy as np
from tqdm import tqdm
import jax

# ### $sin(x)$
# **Task**: Overfit to a 8-bit quantized sin(x) from 0 - 2*Pi -- sampled 360 times.
#
#  @Note: The Feed-Forward model won't necessarily be able to fit this data (optimization is hard)
#  As a sanity check, you can try running with N_CLASSES = 2 (-1, 1) and d_model = 1...
#  this is the simplest "majority rule" experiment => gets 100% test accuracy.
#
#  @Note: RNN & S4 *should* fit this perfectly... but needs to be verified.


def create_sin_x_dataset(n_examples=1024, bsz=128):
    print("[*] Generating Toy Dataset: sin(x)...")

    # Constants
    SEQ_LENGTH, N_CLASSES = 360, 8
    x = np.linspace(0, 2 * np.pi, num=SEQ_LENGTH)
    y = np.digitize(np.sin(x), np.linspace(-1, 1, num=N_CLASSES))

    # Tile this `n_examples` times...
    data = torch.Tensor(
        np.tile(np.expand_dims(np.expand_dims(y, -1), 0), reps=[n_examples, 1, 1])
    )

    # Build Datasets -- Two entries to match (inputs, targets) structure
    train = TensorDataset(data, data)
    test = TensorDataset(data[:1], data[:1])

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(train, batch_size=bsz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=bsz, shuffle=False)

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH


# ### $sin(ax + b)$
# **Task**: Fit arbitrary 8-bit quantized functions of the form sin(ax + b) from 0 - 2*Pi -- sampled 360 times.
#
# In this dataset, `a` controls amplitude and `b` controls phase and are sampled uniformly at random in prespecified
# intervals.


def create_sin_ax_b_dataset(n_examples=20000, bsz=128):
    print("[*] Generating sin(ax + b) Dataset...")

    # Constants – `a` sampled uniform from [1, 10], `b` sampled uniform [0, 5]
    SEQ_LENGTH, N_CLASSES, A_MAX, B_MAX = 360, 8, 10, 5
    train_data, test_data = [], []
    data_key = jax.random.PRNGKey(21)

    # Loop through `n_examples` and generate data
    print(f"\t=>> Generating {n_examples} Training Examples...")
    x = np.linspace(0, 2 * np.pi, num=SEQ_LENGTH)
    for _ in tqdm(range(n_examples)):
        data_key, a_rng, b_rng = jax.random.split(data_key, num=3)

        # Compute a, b
        a, b = jax.random.uniform(a_rng, minval=1.0, maxval=A_MAX), jax.random.uniform(
            b_rng, maxval=B_MAX
        )
        train_data.append(
            np.digitize(np.sin(a * x + b), np.linspace(-1, 1, num=N_CLASSES))
        )

    # Generate 1 Batch of Test Examples
    print(f"\t=>> Generating {bsz} Test Examples...")
    for _ in tqdm(range(bsz)):
        data_key, a_rng, b_rng = jax.random.split(data_key, num=3)

        # Compute a, b
        a, b = jax.random.uniform(a_rng, minval=1.0, maxval=A_MAX), jax.random.uniform(
            b_rng, maxval=B_MAX
        )
        test_data.append(
            np.digitize(np.sin(a * x + b), np.linspace(-1, 1, num=N_CLASSES))
        )

        # Build Datasets - Two entries to match (inputs, targets) structure
        train_data = torch.Tensor(np.expand_dims(np.array(train_data), -1))
        test_data = torch.Tensor(np.expand_dims(np.array(test_data), -1))
        train = TensorDataset(train_data, train_data)
        test = TensorDataset(test_data, test_data)

        # Return data loaders, with the provided batch size
        trainloader = torch.utils.data.DataLoader(train, batch_size=bsz, shuffle=True)
        testloader = torch.utils.data.DataLoader(test, batch_size=bsz, shuffle=False)

        return trainloader, testloader, N_CLASSES, SEQ_LENGTH


# ### MNIST Sequence Modeling
# **Task**: Predict next pixel value given history, in an autoregressive fashion (784 pixels x 256 values).
#
# While we train on full sequences, generations should probably condition on first 10-25% of image.


def create_mnist_dataset(bsz=128):
    print("[*] Generating MNIST Sequence Modeling Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES = 784, 256

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(1, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(train, batch_size=bsz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=bsz, shuffle=False)

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH


Datasets = {
    "MNist": create_mnist_dataset,
    "sin": create_sin_x_dataset,
    "sin_noise": create_sin_ax_b_dataset,
}
