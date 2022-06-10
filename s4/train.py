import os
import shutil
from functools import partial
import hydra
import jax
import jax.numpy as np
import optax
import torch
from flax import linen as nn
from flax.training import checkpoints, train_state
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import Optional
from .data import Datasets
from .dss import DSSLayer
from .s4 import BatchStackedModel, S4Layer, SSMLayer
from .s4d import S4DLayer
from .s4 import sample_mnist_prefix


try:
    # Slightly nonstandard import name to make config easier - see example_train()
    import wandb

    assert hasattr(wandb, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

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
    rng,
    model_cls,
    trainloader,
    lr=1e-3,
    lr_layer=None,
    lr_schedule=False,
    weight_decay=0.0,
    total_steps=-1,
):
    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        np.array(next(iter(trainloader))[0].numpy()),
    )
    # Note: Added immediate `unfreeze()` to play well w/ Optax. See below!
    params = params["params"].unfreeze()

    # Handle learning rates:
    # - LR scheduler
    # - Set custom learning rates on some SSM parameters

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

    if lr_schedule:
        schedule_fn = lambda lr: optax.cosine_onecycle_schedule(
            peak_value=lr,
            transition_steps=total_steps,
            pct_start=0.1,
        )
    else:
        schedule_fn = lambda lr: lr
    # lr_layer is a dictionary from parameter name to LR multiplier
    if lr_layer is None:
        lr_layer = {}

    optimizers = {
        k: optax.adam(learning_rate=schedule_fn(v * lr))
        for k, v in lr_layer.items()
    }
    # Add default optimizer
    # Note: it would be better to use a dummy key such as None that can't conflict with parameter names,
    # but this causes a hard-to-trace error; it seems that the transforms keys list is being sorted inside optax.multi_transform
    # which causes an error since None can't be compared to str
    optimizers["__default__"] = optax.adamw(
        learning_rate=schedule_fn(lr),
        weight_decay=weight_decay,
    )
    name_map = map_nested_fn(lambda k, _: k if k in lr_layer else "__default__")
    tx = optax.multi_transform(optimizers, name_map)
    # For debugging, this would be the default transform with no scheduler or special params
    # tx = optax.adamw(learning_rate=lr, weight_decay=0.01)

    # Check that all special parameter names are actually parameters
    extra_keys = set(lr_layer.keys()) - set(jax.tree_leaves(name_map(params)))
    assert (
        len(extra_keys) == 0
    ), f"Special params {extra_keys} do not correspond to actual params"

    # Print parameter count
    _is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(
        lambda k, param: param.size * (2 if _is_complex(param) else 1)
        if lr_layer.get(k, lr) > 0.0
        else 0
    )(params)
    print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    print(f"[*] Total training steps: {total_steps}")

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


# We also use this opportunity to write generic train_epoch and validation functions. These functions generally
# operate by taking in a training state, model class, dataloader, and critically, the model-specific step function.
# We define the step functions on a model-specific basis below.


def train_epoch(state, rng, model, trainloader, classification=False):
    # Store Metrics
    model = model(training=True)
    batch_losses, batch_accuracies = [], []
    for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())  # Not the most efficient...
        rng, drop_rng = jax.random.split(rng)
        state, loss, acc = train_step(
            state,
            drop_rng,
            inputs,
            labels,
            model,
            classification=classification,
        )
        batch_losses.append(loss)
        batch_accuracies.append(acc)

    # Return average loss over batches
    return (
        state,
        np.mean(np.array(batch_losses)),
        np.mean(np.array(batch_accuracies)),
    )


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

    return np.mean(np.array(losses)), np.mean(np.array(accuracies))


# ### Feed-Forward Model
# Here, we establish a skeleton for a simple, history-blind feed-forward model. For each element $x_t$ of a sequence, our
# feed-forward model attempts to predict $x_{t+1}$. During generation, the predicted "token" is fed as the new current
# element.


class FeedForwardModel(nn.Module):
    d_model: int
    N : int
    l_max: int
    decode: bool = False
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
        logits, mod_vars = model.apply(
            {"params": params},
            batch_inputs,
            rngs={"dropout": rng},
            mutable=["intermediates"],
        )
        loss = np.mean(cross_entropy_loss(logits, batch_labels))
        acc = np.mean(compute_accuracy(logits, batch_labels))
        return loss, (logits, acc)

    if not classification:
        batch_labels = batch_inputs[:, :, 0]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, acc)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc


@partial(jax.jit, static_argnums=(3, 4))
def eval_step(batch_inputs, batch_labels, params, model, classification=False):
    if not classification:
        batch_labels = batch_inputs[:, :, 0]
    logits = model.apply({"params": params}, batch_inputs)
    loss = np.mean(cross_entropy_loss(logits, batch_labels))
    acc = np.mean(compute_accuracy(logits, batch_labels))
    return loss, acc


# ### LSTM Recurrent Model
# Here, we build a simple LSTM sequence model (w/ optional stacked layers). These are fully recurrent
# models, and are initialized with a 0-hidden state, and rolled out for the full sequence length.


class LSTMRecurrentModel(nn.Module):
    N : int
    l_max: int
    d_model: int


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
    "ssm": SSMLayer,
    "s4": S4Layer,
    "dss": DSSLayer,
    "s4d": S4DLayer,
}


def example_train(
    dataset : str,
    layer: str,
    seed : int,
    model : DictConfig,
    train : DictConfig,
):
    # Warnings and sanity checks
    if not train.checkpoint:
        print("[*] Warning: models are not being checkpoint")

    # Set randomness...
    print("[*] Setting Randomness...")
    torch.random.manual_seed(seed) # For dataloader order
    key = jax.random.PRNGKey(seed)
    key, rng, train_rng = jax.random.split(key, num=3)

    # Check if classification dataset
    classification = "classification" in dataset

    # Create dataset
    create_dataset_fn = Datasets[dataset]
    trainloader, testloader, n_classes, l_max, d_input = create_dataset_fn(
        bsz=train.bsz
    )
    in_shape = (train.bsz, l_max, d_input)  # Input shape

    # Get model class and arguments
    layer_cls = Models[layer]
    model.layer.l_max = l_max

    # Extract custom hyperparameters from model class
    lr_layer = getattr(layer_cls, "lr", None)

    print(f"[*] Starting `{layer}` Training on `{dataset}` =>> Initializing...")

    model_cls = partial(
        BatchStackedModel,
        layer_cls=layer_cls,
        d_output=n_classes,
        classification=classification,
        **model,
    )

    state = create_train_state(
        rng,
        model_cls,
        trainloader,
        lr=train.lr,
        lr_layer=lr_layer,
        lr_schedule=train.lr_schedule,
        weight_decay=train.weight_decay,
        total_steps=len(trainloader) * train.epochs,
    )

    # Loop over epochs
    best_loss, best_acc, best_epoch = 10000, 0, 0
    for epoch in range(train.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        state, train_loss, train_acc = train_epoch(
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
            f"\tTrain Loss: {train_loss:.5f} -- Train Accuracy:"
            f" {train_acc:.4f}\n\t Test Loss: {test_loss:.5f} --  Test Accuracy:"
            f" {test_acc:.4f}"
        )

        # Save a checkpoint each epoch & handle best (test loss... not "copacetic" but ehh)
        if train.checkpoint:
            suf = f"-{train.suffix}" if train.suffix is not None else ""
            run_id = f"checkpoints/{dataset}/{layer}-d_model={model.d_model}-lr={train.lr}-bsz={train.bsz}{suf}"
            ckpt_path = checkpoints.save_checkpoint(
                run_id,
                state,
                epoch,
                keep=train.epochs,
            )

            if train.sample is not None:
                if dataset == "mnist":
                    sample_fn = partial(sample_mnist_prefix, length=784, prefix=train.sample)
                else:
                    raise NotImplementedError("Sampling currently only supported for MNIST")

                model_cls = partial(
                    BatchStackedModel,
                    layer_cls=layer_cls,
                    d_output=n_classes,
                    classification=classification,
                    **model,
                )
                samples, examples = sample_fn(run_id, model_cls(decode=True, training=False), rng=rng)
                if wandb is not None:
                    samples = [wandb.Image(sample) for sample in samples]
                    wandb.log({"samples": samples}, commit=False)
                    examples = [wandb.Image(example) for example in examples]
                    wandb.log({"examples": examples}, commit=False)

        if (classification and test_acc > best_acc) or (
            not classification and test_loss < best_loss
        ):
            # Create new "best-{step}.ckpt and remove old one
            if train.checkpoint:
                shutil.copy(ckpt_path, f"{run_id}/best_{epoch}")
                if os.path.exists(f"{run_id}/best_{best_epoch}"):
                    os.remove(f"{run_id}/best_{best_epoch}")

            best_loss, best_acc, best_epoch = test_loss, test_acc, epoch

        # Print best accuracy & loss so far...
        print(
            f"\tBest Test Loss: {best_loss:.5f} -- Best Test Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        if wandb is not None:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "test/loss": test_loss,
                    "test/accuracy": test_acc,
                },
                step=epoch,
            )
            wandb.run.summary["Best Test Loss"] = best_loss
            wandb.run.summary["Best Test Accuracy"] = best_acc
            wandb.run.summary["Best Epoch"] = best_epoch


@hydra.main(version_base=None, config_path="", config_name="config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False) # Allow writing keys

    # Track with wandb
    if wandb is not None:
        wandb_cfg = cfg.pop("wandb")
        wandb.init(
            **wandb_cfg,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    example_train(**cfg)

if __name__ == "__main__":
    main()
