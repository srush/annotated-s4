import os
import jax
import numpy as np
import torch
import torchtext
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset, DatasetDict
from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm

## lra imdb
def create_imdb_classification_dataset(bsz=128):
    # Constants, the default max length is 4096
    APPEND_BOS = False
    APPEND_EOS = True
    LOAD_WORDER = 20
    MIN_FREQ = 15

    SEQ_LENGTH, N_CLASSES, IN_DIM = 2048, 2, 1

    # load data using huggingface datasets
    dataset = load_dataset("imdb")
    dataset = DatasetDict(train=dataset["train"], test=dataset["test"])

    l_max = SEQ_LENGTH - int(APPEND_BOS) - int(APPEND_EOS)
    # step one, byte level tokenization
    tokenize = lambda example: {"tokens": list(example["text"])[:l_max]}
    dataset = dataset.map(
        tokenize,
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

    numericalize = lambda example: {
        "input_ids": vocab(
            (["<bos>"] if APPEND_BOS else [])
            + example["tokens"]
            + (["<eos>"] if APPEND_EOS else [])
        )
    }
    dataset = dataset.map(
        numericalize,
        remove_columns=["tokens"],
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=max(LOAD_WORDER, 1),
    )

    # print("numericalize result for first example:", dataset['train']['input_ids'][0])

    dataset['train'].set_format(type='torch', columns=['input_ids', 'label'])
    dataset['test'].set_format(type='torch', columns=['input_ids', 'label'])

    def imdb_collate(batch):
        batchfy_input_ids = [data["input_ids"] for data in batch]
        batchfy_labels = torch.cat([data["label"].unsqueeze(0) for data in batch], dim=0)
        batchfy_input_ids = torch.nn.utils.rnn.pad_sequence(
            batchfy_input_ids + [torch.zeros(SEQ_LENGTH)], padding_value=vocab["<pad>"], batch_first=True
        )
        return batchfy_input_ids[:-1], batchfy_labels

    trainloader = torch.utils.data.DataLoader(
        dataset['train'], batch_size=bsz, shuffle=True, collate_fn=imdb_collate)

    testloader = torch.utils.data.DataLoader(
        dataset['test'], batch_size=bsz, shuffle=True, collate_fn=imdb_collate)

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM

## lra listops
def create_listops_classification_dataset(bsz):
    # global constants, default maximal length is 2048
    list_dir = "listops-1000"
    APPEND_BOS = False
    APPEND_EOS = True
    LOAD_WORDER = 4
    SEQ_LENGTH, N_CLASSES, IN_DIM = 2048, 10, 1

    #  tokenizer
    def listops_tokenizer(s):
        return s.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()

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
    tokenize = lambda example: {"tokens": tokenizer(example["Source"])[:l_max]}

    dataset = dataset.map(
        tokenize,
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

    # step 3, numerialize
    vocab.set_default_index(vocab["<unk>"])

    numericalize = lambda example: {
        "input_ids": vocab(
            (["<bos>"] if APPEND_BOS else [])
            + example["tokens"]
            + (["<eos>"] if APPEND_EOS else [])
        )
    }
    dataset = dataset.map(
        numericalize,
        remove_columns=["tokens"],
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=max(LOAD_WORDER, 1),
    )

    # print("Check the numerical results:", len(dataset['train']['input_ids']), dataset['train']['input_ids'][0])

    # training and test formats here
    dataset['train'].set_format(type='torch', columns=['input_ids', 'Target'])
    dataset['test'].set_format(type='torch', columns=['input_ids', 'Target'])

    # batchfy for training
    def listops_collate(batch):
        batchfy_input_ids = [data["input_ids"] for data in batch]
        batchfy_labels = torch.cat([data["Target"].unsqueeze(0) for data in batch], dim=0)
        batchfy_input_ids = torch.nn.utils.rnn.pad_sequence(
            batchfy_input_ids + [torch.zeros(SEQ_LENGTH)], padding_value=vocab["<pad>"], batch_first=True
        )
        return batchfy_input_ids[:-1], batchfy_labels

    trainloader = torch.utils.data.DataLoader(
        dataset['train'], batch_size=bsz, shuffle=True, collate_fn=listops_collate)

    testloader = torch.utils.data.DataLoader(
        dataset['test'], batch_size=bsz, shuffle=True, collate_fn=listops_collate)

    return trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM

Datasets = {
    "imdb-classification": create_imdb_classification_dataset,
    "listops-classification":create_listops_classification_dataset
}
