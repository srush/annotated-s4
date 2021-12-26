from .train import example_train
from .data import Datasets
from .s4 import S4LayerInit, NaiveSSMInit


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=Datasets.keys(), required=True)
    args = parser.parse_args()

    # InitS4Layer(64)
    # model = NaiveSSMInit(64)
    model = S4LayerInit(N=64)
    example_train(model, Datasets[args.dataset], epochs=100)
