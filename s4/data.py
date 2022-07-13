import os
import jax
import numpy as np
import torch
import torchtext
import torchvision
import torchvision.transforms as transforms
from datasets import DatasetDict, load_dataset
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
        np.tile(
            np.expand_dims(np.expand_dims(y, -1), 0), reps=[n_examples, 1, 1]
        )
    )

    # Build Datasets -- Two entries to match (inputs, targets) structure
    train = TensorDataset(data, data)
    test = TensorDataset(data[:1], data[:1])

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False
    )

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
        a, b = jax.random.uniform(
            a_rng, minval=1.0, maxval=A_MAX
        ), jax.random.uniform(b_rng, maxval=B_MAX)
        train_data.append(
            np.digitize(np.sin(a * x + b), np.linspace(-1, 1, num=N_CLASSES))
        )

    # Generate 1 Batch of Test Examples
    print(f"\t=>> Generating {bsz} Test Examples...")
    for _ in tqdm(range(bsz)):
        data_key, a_rng, b_rng = jax.random.split(data_key, num=3)

        # Compute a, b
        a, b = jax.random.uniform(
            a_rng, minval=1.0, maxval=A_MAX
        ), jax.random.uniform(b_rng, maxval=B_MAX)
        test_data.append(
            np.digitize(np.sin(a * x + b), np.linspace(-1, 1, num=N_CLASSES))
        )

    # Build Datasets - Two entries to match (inputs, targets) structure
    train_data = torch.Tensor(np.expand_dims(np.array(train_data), -1))
    test_data = torch.Tensor(np.expand_dims(np.array(test_data), -1))
    train = TensorDataset(train_data, train_data)
    test = TensorDataset(test_data, test_data)

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### MNIST Sequence Modeling
# **Task**: Predict next pixel value given history, in an autoregressive fashion (784 pixels x 256 values).
#
def create_mnist_dataset(bsz=128):
    print("[*] Generating MNIST Sequence Modeling Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 256, 1

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (x.view(IN_DIM, SEQ_LENGTH).t() * 255).int()
            ),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train,
        batch_size=bsz,
        shuffle=True,
    )
    testloader = torch.utils.data.DataLoader(
        test,
        batch_size=bsz,
        shuffle=False,
    )

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
            "\tDownloading Simplified Drawings from Google Cloud (will take a"
            " while)..."
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
        data, labels = np.concatenate(data, axis=0), np.concatenate(
            labels, axis=0
        )

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
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### FSDD Sequence Modeling
# **Task**: Predict next wav value given history, in an autoregressive fashion (6400 pixels x 256 values).
#
def create_fsdd_dataset(bsz=128):
    print("[*] Generating FSDD Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 6400, 256, 1

    from torchaudio.transforms import MuLawEncoding
    from torchfsdd import TorchFSDDGenerator, TrimSilence

    # Create a transformation pipeline to apply to the recordings
    tf = transforms.Compose(
        [
            TrimSilence(threshold=1e-6),
            MuLawEncoding(quantization_channels=255),
            transforms.Lambda(
                lambda x: torch.nn.functional.pad(
                    x.view(-1), (0, SEQ_LENGTH - x.shape[0]), "constant", 255
                ).view(-1, 1)
            ),
        ]
    )

    # Fetch the latest version of FSDD and initialize a generator with those files
    fsdd = TorchFSDDGenerator("local", "recordings/", transforms=tf)

    # Create two Torch datasets for a train-test split from the generator
    train, test = fsdd.train_test_split(test_size=0.1)

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### Speech Commands Sequence Modeling
# **Task**: Predict next wav value given history, in an autoregressive fashion (8000 samples x 256 values).
#
def create_sc_dataset(bsz=128):
    print("[*] Generating SC Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 8000, 256, 1
    import os
    from torchaudio.datasets import SPEECHCOMMANDS
    from torchaudio.transforms import MuLawEncoding, Resample

    # # Create a transformation pipeline to apply to the recordings
    tf = transforms.Compose(
        [
            Resample(16000, SEQ_LENGTH),
            MuLawEncoding(quantization_channels=255),
            transforms.Lambda(
                lambda x: torch.nn.functional.pad(
                    x.view(-1),
                    (0, SEQ_LENGTH - x.view(-1).shape[0]),
                    "constant",
                    255,
                ).view(-1, 1)
            ),
        ]
    )

    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, subset: str = None):
            super().__init__("./", download=True)
            digits = [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            ]

            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    return [
                        os.path.join(self._path, line.strip())
                        for line in fileobj
                        if line.split("/")[0] in digits
                    ]

            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list(
                    "testing_list.txt"
                )
                excludes = set(excludes)
                self._walker = [
                    w
                    for w in self._walker
                    if w not in excludes
                    if w.split("/")[-2] in digits
                ]

        def __getitem__(self, n):
            (
                waveform,
                sample_rate,
                label,
                speaker_id,
                utterance_number,
            ) = super().__getitem__(n)
            out = tf(waveform)
            return out, 0

    # Create training and testing split of the data. We do not use validation in this tutorial.
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")

    waveform, label = train_set[0]
    print(waveform.shape, label)
    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=bsz, shuffle=False
    )

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
            transforms.Normalize(mean=0.5, std=0.5),
            transforms.Lambda(lambda x: x.view(IN_DIM, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False
    )

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
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
            transforms.Lambda(lambda x: x.view(IN_DIM, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# ### FSDD Classification
# **Task**: Predict FSDD class given sequence model over pixels (6400 wav => 10 classes).
def create_fsdd_classification_dataset(bsz=128):
    print("[*] Generating FSDD Classification Dataset...")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 6400, 10, 1

    from torchaudio.transforms import MuLawEncoding
    from torchfsdd import TorchFSDDGenerator, TrimSilence

    # Create a transformation pipeline to apply to the recordings
    tf = transforms.Compose(
        [
            TrimSilence(threshold=1e-6),
            MuLawEncoding(quantization_channels=512),
            transforms.Lambda(
                lambda x: torch.nn.functional.pad(
                    x, (0, 6400 - x.shape[0])
                ).view(-1, 1)
            ),
        ]
    )

    # Fetch the latest version of FSDD and initialize a generator with those files
    fsdd = TorchFSDDGenerator(version="master", transforms=tf)

    # Create two Torch datasets for a train-test split from the generator
    train, test = fsdd.train_test_split(test_size=0.1)

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


def create_imdb_classification_dataset(bsz=128):
    # Constants, the default max length is 4096
    APPEND_BOS = False
    APPEND_EOS = True
    LOAD_WORDER = 20
    MIN_FREQ = 15

    SEQ_LENGTH, N_CLASSES, IN_DIM = 2048, 2, 135

    # load data using huggingface datasets
    dataset = load_dataset("imdb")
    dataset = DatasetDict(train=dataset["train"], test=dataset["test"])

    l_max = SEQ_LENGTH - int(APPEND_BOS) - int(APPEND_EOS)

    # step one, byte level tokenization
    dataset = dataset.map(
        lambda example: {"tokens": list(example["text"])[:l_max]},
        remove_columns=["text"],
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=max(LOAD_WORDER, 1),
    )

    # print("byte characters for first example:", dataset['train']['tokens'][0])

    # step two, build vocabulary based on the byte characters, each character appear at least MIN_FREQ times
    vocab = torchtext.vocab.build_vocab_from_iterator(
        dataset["train"]["tokens"],
        min_freq=MIN_FREQ,
        specials=(
            ["<pad>", "<unk>"]
            + (["<bos>"] if APPEND_BOS else [])
            + (["<eos>"] if APPEND_EOS else [])
        ),
    )

    # step three, numericalize the tokens
    vocab.set_default_index(vocab["<unk>"])

    dataset = dataset.map(
        lambda example: {
            "input_ids": vocab(
                (["<bos>"] if APPEND_BOS else [])
                + example["tokens"]
                + (["<eos>"] if APPEND_EOS else [])
            )
        },
        remove_columns=["tokens"],
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=max(LOAD_WORDER, 1),
    )

    # print("numericalize result for first example:", dataset['train']['input_ids'][0])

    dataset["train"].set_format(type="torch", columns=["input_ids", "label"])
    dataset["test"].set_format(type="torch", columns=["input_ids", "label"])

    def imdb_collate(batch):
        batchfy_input_ids = [data["input_ids"] for data in batch]
        batchfy_labels = torch.cat(
            [data["label"].unsqueeze(0) for data in batch], dim=0
        )
        batchfy_input_ids = torch.nn.utils.rnn.pad_sequence(
            batchfy_input_ids + [torch.zeros(SEQ_LENGTH)],
            padding_value=vocab["<pad>"],
            batch_first=True,
        )
        batchfy_input_ids = torch.nn.functional.one_hot(
            batchfy_input_ids[:-1], IN_DIM
        )
        return batchfy_input_ids, batchfy_labels

    trainloader = torch.utils.data.DataLoader(
        dataset["train"], batch_size=bsz, shuffle=True, collate_fn=imdb_collate
    )

    testloader = torch.utils.data.DataLoader(
        dataset["test"], batch_size=bsz, shuffle=True, collate_fn=imdb_collate
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


# listops
def create_listops_classification_dataset(bsz):
    # global constants, default maximal length is 2048
    list_dir = "listops-1000"
    APPEND_BOS = False
    APPEND_EOS = True
    LOAD_WORDER = 20
    SEQ_LENGTH, N_CLASSES, IN_DIM = 2048, 10, 20

    #  tokenizer
    def listops_tokenizer(s):
        return s.translate(
            {ord("]"): ord("X"), ord("("): None, ord(")"): None}
        ).split()

    # step 1, load and build datasets
    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(f"{list_dir}/basic_train.tsv"),
            "val": str(f"{list_dir}/basic_val.tsv"),
            "test": str(f"{list_dir}/basic_test.tsv"),
        },
        delimiter="\t",
        keep_in_memory=True,
    )

    tokenizer = listops_tokenizer
    l_max = SEQ_LENGTH - int(APPEND_BOS) - int(APPEND_EOS)

    dataset = dataset.map(
        lambda example: {"tokens": tokenizer(example["Source"])[:l_max]},
        remove_columns=["Source"],
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=max(LOAD_WORDER, 1),
    )

    # step 2, build vocabulary
    vocab = torchtext.vocab.build_vocab_from_iterator(
        dataset["train"]["tokens"],
        specials=(
            ["<pad>", "<unk>"]
            + (["<bos>"] if APPEND_BOS else [])
            + (["<eos>"] if APPEND_EOS else [])
        ),
    )

    # step 3, numericalize
    vocab.set_default_index(vocab["<unk>"])

    dataset = dataset.map(
        lambda example: {
            "input_ids": vocab(
                (["<bos>"] if APPEND_BOS else [])
                + example["tokens"]
                + (["<eos>"] if APPEND_EOS else [])
            )
        },
        remove_columns=["tokens"],
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=max(LOAD_WORDER, 1),
    )

    # print("Check the numerical results:", len(dataset['train']['input_ids']), dataset['train']['input_ids'][0])

    # training and test formats here
    dataset["train"].set_format(type="torch", columns=["input_ids", "Target"])
    dataset["test"].set_format(type="torch", columns=["input_ids", "Target"])

    # batchfy for training
    def listops_collate(batch):
        batchfy_input_ids = [data["input_ids"] for data in batch]
        batchfy_labels = torch.cat(
            [data["Target"].unsqueeze(0) for data in batch], dim=0
        )
        batchfy_input_ids = torch.nn.utils.rnn.pad_sequence(
            batchfy_input_ids + [torch.zeros(SEQ_LENGTH)],
            padding_value=vocab["<pad>"],
            batch_first=True,
        )
        batchfy_input_ids = torch.nn.functional.one_hot(
            batchfy_input_ids[:-1], IN_DIM
        )  # one hot encoding for the input
        return batchfy_input_ids, batchfy_labels

    trainloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=bsz,
        shuffle=True,
        collate_fn=listops_collate,
    )

    testloader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=bsz,
        shuffle=True,
        collate_fn=listops_collate,
    )

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


Datasets = {
    "mnist": create_mnist_dataset,
    "quickdraw": create_quickdraw_dataset,
    "fsdd": create_fsdd_dataset,
    "sc": create_sc_dataset,
    "sin": create_sin_x_dataset,
    "sin_noise": create_sin_ax_b_dataset,
    "mnist-classification": create_mnist_classification_dataset,
    "fsdd-classification": create_fsdd_classification_dataset,
    "cifar-classification": create_cifar_classification_dataset,
    "imdb-classification": create_imdb_classification_dataset,
    "listops-classification": create_listops_classification_dataset,
}
