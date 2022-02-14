import os
import shutil
from functools import partial
import jax
import jax.numpy as np
import optax
from flax import linen as nn
from flax.training import checkpoints, train_state
from tqdm import tqdm
from .data import Datasets
from .s4 import BatchSeqModel, S4LayerInit, SSMInit, make_NPLR_HiPPO, discretize


# ## Baseline Models
#
# We start with definitions of various models we're already familiar with, starting with a feed-forward
# (history-blind) projection model, followed by a strong LSTM-based recurrent baseline.

# ### Utilities
# We define a couple of utility functions below to compute a standard cross-entropy loss, and compute
# "token"-level prediction accuracy.


@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label


# As we're using Flax, we also write a utility function to return a default TrainState object.
# This function initializes model parameters, as well as our optimizer. Note that for S4 models,
# we use a custom learning rate for parameters of the S4 kernel (lr = 0.001, no weight decay).
def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(
    model_name,
    model_cls,
    rng,
    in_dim=1,
    bsz=128,
    seq_len=784,
    lr=1e-3,
    lr_schedule=False,
    total_steps=-1,
):
    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        np.ones((bsz, seq_len, in_dim)),
    )[
        "params"
    ].unfreeze()  # Note: Added immediate `unfreeze()` to play well w/ Optax. See below!

    # Implement LR Schedule (No change for first 30% of training, then decay w/ cubic polynomial to 0 for last 70%)
    if lr_schedule:
        lr = optax.polynomial_schedule(
            init_value=lr,
            end_value=0.0,
            power=3,
            transition_begin=int(0.3 * total_steps),
            transition_steps=int(0.7 * total_steps),
        )

    # # S4 uses a Fixed LR = 1e-3 with NO weight decay for the S4 Matrices, higher LR elsewhere
    if "s4" in model_name:
        # Note for Debugging... this is all undocumented and so weird. The following links are helpful...
        #
        #   > Flax "Recommended" interplay w/ Optax (this bridge needs ironing):
        #       https://github.com/google/flax/blob/main/docs/flip/1009-optimizer-api.md#multi-optimizer
        #
        #   > But... masking doesn't work like the above example suggests!
        #       Root Explanation: https://github.com/deepmind/optax/issues/159
        #       Fix: https://github.com/deepmind/optax/discussions/167
        #
        #   > Also... Flax FrozenDict doesn't play well with rest of Jax + Optax...
        #       https://github.com/deepmind/optax/issues/160#issuecomment-896460796
        #
        #   > Solution: Use Optax.multi_transform!
        s4_fn = map_nested_fn(
            lambda k, _: 
            "s4"
            if k in ["B"]
            else ("none" if k in ["D", "log_step"]
                  else "regular")
        )
        tx = optax.multi_transform(
            {
                "none": optax.sgd(learning_rate=0.0),
                "s4": optax.adam(learning_rate=1e-3),
                "regular": optax.adamw(learning_rate=lr, weight_decay=0.01),
            },
            s4_fn,
        )

    else:
        tx = optax.adamw(learning_rate=lr, weight_decay=0.01)

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


# We also use this opportunity to write generic train_epoch and validation functions. These functions generally
# operate by taking in a training state, model class, dataloader, and critically, the model-specific step function.
# We define the step functions on a model-specific basis below.


def train_epoch(state, rng, model, trainloader, classification=False):
    # Store Metrics
    model = model(training=True)
    batch_losses = []
    for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())  # Not the most efficient...
        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(
            state,
            drop_rng,
            inputs,
            labels,
            model,
            classification=classification,
        )
        batch_losses.append(loss)

    # Return average loss over batches
    return state, np.mean(np.array(batch_losses))


def validate(params, model, testloader, classification=False):
    # Compute average loss & accuracy
    model = model(training=False)
    losses, accuracies = [], []
    for batch_idx, (inputs, labels) in enumerate(tqdm(testloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())  # Not the most efficient...
        loss, acc = eval_step(
            inputs, labels, params, model, classification=classification
        )
        losses.append(loss)
        accuracies.append(acc)

    # Sampling autoregressively prompted w/ first 100 "tokens"...
    #   => TODO @Sidd
    return np.mean(np.array(losses)), np.mean(np.array(accuracies))


# ### Feed-Forward Model
# Here, we establish a skeleton for a simple, history-blind feed-forward model. For each element $x_t$ of a sequence, our
# feed-forward model attempts to predict $x_{t+1}$. During generation, the predicted "token" is fed as the new current
# element.


class FeedForwardModel(nn.Module):
    d_model: int
    l_max: int

    def setup(self):
        self.dense = nn.Dense(self.d_model)

    def __call__(self, x):
        """x - L x N"""
        return nn.relu(self.dense(x))


# We define separate step functions for running training and evaluation steps, accordingly. These step functions are
# each wrapped in a call to `@jax.jit` which fuses operations, generally leading to high performance gains. These @jit
# calls will become increasingly important as we optimize S4.


@partial(jax.jit, static_argnums=(4, 5))
def train_step(
    state, rng, batch_inputs, batch_labels, model, classification=False
):
    def loss_fn(params):
        if classification:
            logits, mod_vars = model.apply(
                {"params": params},
                batch_inputs,
                rngs={"dropout": rng},
                mutable=["intermediates"],
            )
            loss = np.mean(cross_entropy_loss(logits, batch_labels))
        else:
            logits, mod_vars = model.apply(
                {"params": params},
                batch_inputs[:, :-1],
                rngs={"dropout": rng},
                mutable=["intermediates"],
            )
            loss = np.mean(cross_entropy_loss(logits, batch_inputs[:, 1:, 0]))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@partial(jax.jit, static_argnums=(3, 4))
def eval_step(batch_inputs, batch_labels, params, model, classification=False):
    if classification:
        logits = model.apply({"params": params}, batch_inputs)
        loss = np.mean(cross_entropy_loss(logits, batch_labels))
        acc = np.mean(compute_accuracy(logits, batch_labels))
    else:
        logits = model.apply({"params": params}, batch_inputs[:, :-1])
        loss = np.mean(cross_entropy_loss(logits, batch_inputs[:, 1:, 0]))
        acc = np.mean(compute_accuracy(logits, batch_inputs[:, 1:, 0]))
    return loss, acc


# ### LSTM Recurrent Model
# Here, we build a simple LSTM sequence model (w/ optional stacked layers). These are fully recurrent
# models, and are initialized with a 0-hidden state, and rolled out for the full sequence length.


class LSTMRecurrentModel(nn.Module):
    d_model: int
    l_max: int

    def setup(self):
        LSTM = nn.scan(
            nn.OptimizedLSTMCell,
            in_axes=0,
            out_axes=0,
            variable_broadcast="params",
            split_rngs={"params": False},
        )
        dummy_rng = jax.random.PRNGKey(0)
        self.init_h = nn.OptimizedLSTMCell.initialize_carry(
            dummy_rng, (), self.d_model
        )
        self.LSTM = LSTM(name="lstm_cell")

    def __call__(self, xs):
        return self.LSTM(self.init_h, xs)[1]


# ## Sanity Checks
# Here we provide examples for training & evaluation our baseline models on the various datasets.

Models = {
    "ff": FeedForwardModel,
    "lstm": LSTMRecurrentModel,
    "ssm-naive": SSMInit,
    "s4": S4LayerInit,
}


def example_train(
    model,
    dataset,
    d_model=128,
    bsz=128,
    epochs=10,
    ssm_n=64,
    lr=1e-3,
    lr_schedule=False,
    n_layers=4,
    p_dropout=0.2,
    suffix=None,
):
    # Set randomness...
    print("[*] Setting Randomness...")
    key = jax.random.PRNGKey(0)
    key, rng, train_rng = jax.random.split(key, num=3)

    # Get model class and dataset creation function
    create_dataset_fn = Datasets[dataset]
    if model in ["ssm-naive", "s4"]:
        model_cls = Models[model](N=ssm_n)
    else:
        model_cls = Models[model]

    # Check if classification dataset
    classification = "classification" in dataset

    # Create dataset...
    trainloader, testloader, n_classes, seq_len, in_dim = create_dataset_fn(
        bsz=bsz
    )
    print(f"[*] Starting `{model}` Training on `{dataset}` =>> Initializing...")

    model_cls = partial(
        BatchSeqModel,
        layer=model_cls,
        d_model=d_model,
        d_output=n_classes,
        dropout=p_dropout,
        n_layers=n_layers,
        l_max=seq_len if classification else seq_len - 1,
        classification=classification,
    )
    state = create_train_state(
        model,
        model_cls,
        rng,
        in_dim=in_dim,
        bsz=bsz,
        seq_len=seq_len if classification else seq_len - 1,
        lr=lr,
        lr_schedule=lr_schedule,
        total_steps=len(trainloader) * epochs,
    )

    # Loop over epochs
    best_loss, best_acc, best_epoch = 10000, 0, 0
    for epoch in range(epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        state, train_loss = train_epoch(
            state,
            train_rng,
            model_cls,
            trainloader,
            classification=classification,
        )

        print(f"[*] Running Epoch {epoch + 1} Validation...")
        test_loss, test_acc = validate(
            state.params, model_cls, testloader, classification=classification
        )

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(
            f"\tTrain Loss: {train_loss:.5f} -- Test Loss: {test_loss:.5f} --"
            f" Test Accuracy: {test_acc:.4f}"
        )

        # Save a checkpoint each epoch & handle best (test loss... not "copacetic" but ehh)
        run_id = f"checkpoints/{dataset}/{model}-d_model={d_model}" + (
            f"-{suffix}" if suffix is not None else ""
        )
        ckpt_path = checkpoints.save_checkpoint(
            run_id,
            state,
            epoch,
            keep=epochs,
        )
        if (classification and test_acc > best_acc) or (
            not classification and test_loss < best_loss
        ):
            # Create new "best-{step}.ckpt and remove old one
            shutil.copy(ckpt_path, f"{run_id}/best_{epoch}")
            if os.path.exists(f"{run_id}/best_{best_epoch}"):
                os.remove(f"{run_id}/best_{best_epoch}")

            best_loss, best_acc, best_epoch = test_loss, test_acc, epoch

        # Print best accuracy & loss so far...
        print(
            f"\tBest Test Loss: {best_loss:.5f} -- Best Test Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=Datasets.keys(), required=True
    )
    parser.add_argument(
        "--model", type=str, choices=Models.keys(), required=True
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--suffix", type=str, default=None)

    # Model Parameters
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--p_dropout", type=float, default=0.2)

    # S4 Specific Parameters
    parser.add_argument("--ssm_n", type=int, default=64)

    # Optimization Parameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_schedule", default=False, action="store_true")

    args = parser.parse_args()

    example_train(
        args.model,
        args.dataset,
        epochs=args.epochs,
        d_model=args.d_model,
        bsz=args.bsz,
        ssm_n=args.ssm_n,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        n_layers=args.n_layers,
        p_dropout=args.p_dropout,
        suffix=args.suffix,
    )
