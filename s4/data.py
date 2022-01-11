import os
import jax
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm


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
    SEQ_LENGTH, N_CLASSES, IN_DIM = 16, 8, 1
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

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### $sin(ax + b)$
# **Task**: Fit arbitrary 8-bit quantized functions of the form sin(ax + b) from 0 - 2*Pi -- sampled 360 times.
#
# In this dataset, `a` controls amplitude and `b` controls phase and are sampled uniformly at random in prespecified
# intervals.
def create_sin_ax_b_dataset(n_examples=20000, bsz=128):
    print("[*] Generating sin(ax + b) Dataset...")

    # Constants – `a` sampled uniform from [1, 10], `b` sampled uniform [0, 5]
    SEQ_LENGTH, N_CLASSES, IN_DIM, A_MAX, B_MAX = 16000, 8, 1, 10, 5
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

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### MNIST Sequence Modeling
# **Task**: Predict next pixel value given history, in an autoregressive fashion (784 pixels x 256 values).
#
# While we train on full sequences, generations should probably condition on first 10-25% of image.
def create_mnist_dataset(bsz=128):
    print("[*] Generating MNIST Sequence Modeling Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 256, 1

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x.view(1, SEQ_LENGTH).t() * 256).int()),
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

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### QuickDraw Drawing Generation
# **Task**: Given dataset of <50M Google QuickDraw Sketches as 28 x 28 grayscale values, predict next pixel in an
# autoregressive fashion.
#
# Similar to MNIST Sequence modeling, generations should probably condition on first 10-25% of image. Future work
# should look at modeling drawings at the *stroke* level, present a more natural "interactive" completion aspect for
# folks to play around with!
def create_quickdraw_dataset(bsz=128):
    print("[*] Generating QuickDraw Sequence Modeling Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 256, 1

    if not os.path.exists("data/quickdraw/npy"):
        # Create Dataset
        os.makedirs("data/quickdraw/npy")

        # Note - requires downloading from Google Cloud Bucket; dependency google-cloud-storage installed!
        from google.cloud import storage

        # Download all of the .npy "simplified" drawings...
        print(
            "\tDownloading Simplified Drawings from Google Cloud (will take a while)..."
        )
        client = storage.Client.create_anonymous_client()
        bucket = client.get_bucket("quickdraw_dataset")
        blobs = bucket.list_blobs(prefix="full/numpy_bitmap")
        for b in tqdm(list(blobs)):
            b.download_to_filename(
                f"data/quickdraw/npy/{b.name.split('/')[-1].lower()}"
            )

    # Iterate through Dataset, build full set
    if os.path.exists("data/quickdraw/data.npz"):
        print("\tLoading Full Dataset from npz file (may take a bit)...")
        npz = np.load("data/quickdraw/data.npz")
        data, labels = npz["data"], npz["labels"]
    else:
        print("\tTensorizing Dataset (will also take a while)...")
        data, labels = [], []
        for i, c_name in enumerate(tqdm(os.listdir("data/quickdraw/npy"))):
            class_data = np.load(f"data/quickdraw/npy/{c_name}")
            data.append(class_data)
            labels.append(np.ones(len(class_data)) * i)

        # Create "full" dataset & labels
        data, labels = np.concatenate(data, axis=0), np.concatenate(labels, axis=0)

        # Save Dataset
        np.savez("data/quickdraw/data.npz", data=data, labels=labels)

    # Generate train/test splits... test should be a fraction of 0.001 of total set (assuming in 10s of millions)
    print("\tGenerating Train/Test Splits...")
    data, labels, n_test = (
        torch.Tensor(data),
        torch.Tensor(labels),
        int(0.001 * len(data)),
    )
    dataset = TensorDataset(data.unsqueeze(-1), labels)
    train, test = random_split(
        dataset, [len(data) - n_test, n_test], torch.Generator().manual_seed(3)
    )

    # Return data loaders with the provided batch size
    trainloader = torch.utils.data.DataLoader(train, batch_size=bsz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=bsz, shuffle=False)

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### MNIST Classification
# **Task**: Predict MNIST class given sequence model over pixels (784 pixels => 10 classes).
def create_mnist_classification_dataset(bsz=128):
    print("[*] Generating MNIST Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 10, 1
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x.view(1, SEQ_LENGTH).t() * 256).int()),
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

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### CIFAR-10 Classification
# **Task**: Predict CIFAR-10 class given sequence model over pixels (32 x 32 x 3 RGB image => 10 classes).
def create_cifar_classification_dataset(bsz=128):
    print("[*] Generating CIFAR-10 Classification Dataset")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 32 * 32, 10, 3
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(3, 1024).t()),
        ]
    )

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(train, batch_size=bsz, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=bsz, shuffle=False)

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


Datasets = {
    "mnist": create_mnist_dataset,
    "quickdraw": create_quickdraw_dataset,
    "sin": create_sin_x_dataset,
    "sin_noise": create_sin_ax_b_dataset,
    "mnist-classification": create_mnist_classification_dataset,
    "cifar-classification": create_cifar_classification_dataset,
}
