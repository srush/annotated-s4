from functools import partial
import jax
import jax.numpy as np
import optax
from flax import linen as nn
from flax.training import train_state, checkpoints

from tqdm import tqdm
from .data import Datasets


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
# This function initializes model parameters, as well as our optimizer.


def create_train_state(model, rng, bsz=128, seq_len=784, lr=1e-3):
    model = model(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    params = model.init(
        {"params": init_rng, "dropout": dropout_rng}, np.ones((bsz, seq_len - 1, 1))
    )["params"]
    tx = optax.adamw(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# We also use this opportunity to write generic train_epoch and validation functions. These functions generally
# operate by taking in a training state, model class, dataloader, and critically, the model-specific step function.
# We define the step functions on a model-specific basis below.


def train_epoch(state, rng, model, trainloader):
    # Store Metrics
    model = model(training=True)
    batch_losses = []
    for batch_idx, (inputs, _) in enumerate(tqdm(trainloader)):
        inputs = np.array(inputs.numpy())
        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(state, drop_rng, inputs, model)
        batch_losses.append(loss)

    # Return average loss over batches
    return state, np.mean(np.array(batch_losses))


def validate(params, model, testloader):
    # Compute average loss & accuracy
    model = model(training=False)
    losses, accuracies = [], []
    for batch_idx, (inputs, _) in enumerate(tqdm(testloader)):
        inputs = np.array(inputs.numpy())
        loss, acc = eval_step(inputs, params, model)
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
    l_max : int
    def setup(self):
        self.dense = nn.Dense(self.d_model)

    def __call__(self, x):
        "x - L x N"
        return nn.relu(self.dense(x))


# We define separate step functions for running training and evaluation steps, accordingly. These step functions are
# each wrapped in a call to `@jax.jit` which fuses operations, generally leading to high performance gains. These @jit
# calls will become increasingly important as we optimize S4.


@partial(jax.jit, static_argnums=(3,))
def train_step(state, rng, batch, model):
    def loss_fn(params):
        logits, mod_vars = model.apply({"params": params}, batch[:, :-1], rngs={"dropout": rng},
                                       mutable=["intermediates"])
        loss = np.mean(cross_entropy_loss(logits, batch[:, 1:, 0]))
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # print(state.params)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@partial(jax.jit, static_argnums=(2,))
def eval_step(batch, params, model):
    logits = model.apply({"params": params}, batch[:, :-1])
    loss = np.mean(cross_entropy_loss(logits, batch[:, 1:, 0]))
    acc = np.mean(compute_accuracy(logits, batch[:, 1:, 0]))
    return loss, acc


# ### LSTM Recurrent Model
# Here, we build a simple LSTM sequence model (w/ optional stacked layers). These are fully recurrent
# models, and are initialized with a 0-hidden state, and rolled out for the full sequence length.


class LSTMRecurrentModel(nn.Module):
    d_model: int
    l_max : int
    def setup(self):
        LSTM = nn.scan(
            nn.OptimizedLSTMCell,
            in_axes=0,
            out_axes=0,
            variable_broadcast="params",
            split_rngs={"params": False},
        )
        dummy_rng = jax.random.PRNGKey(0)
        self.init_h = nn.OptimizedLSTMCell.initialize_carry(dummy_rng, (), self.d_model)
        self.LSTM = LSTM(name="lstm_cell")

    def __call__(self, xs):
        return self.LSTM(self.init_h, xs)[1]




    
# General Skeleton for residual Sequence model with  --> takes an sequence layer
class SeqModel(nn.Module):
    layer: nn.Module
    d_output: int
    d_model: int
    l_max : int
    n_layers: int
    dropout: float = 0.2
    training: bool = True
    sampling: bool = False
    
    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        self.layers = tuple(
            [
                (
                    self.layer(d_model=self.d_model, l_max=self.l_max),
                    nn.LayerNorm(),
                    nn.Dense(self.d_model),
                    nn.Dense(self.d_model),
                    nn.Dropout(self.dropout, deterministic=not self.training),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x):
        # x - L x N
        x = self.encoder(x)
        for l, (layer, norm, inp, out, dropout) in enumerate(self.layers):
            x2 = layer(inp(x))
            z = dropout(out(nn.gelu(x2)))
            # z = x2
            x = norm(z + x)

        x = self.decoder(x)
        return nn.log_softmax(x)

BatchSeqModel = nn.vmap(
        SeqModel,
        in_axes=0,
        out_axes=0,
        variable_axes={"params": None, "dropout": None},
        split_rngs={"params": False, "dropout": False},
    )

# ## Sanity Checks
# Here we provide examples for training & evaluation our baseline models on the various datasets.


def example_train(
    model_cls,
    create_dataset_fn,
    bsz=128,
    epochs=10,
):
    # Set randomness...
    print("[*] Setting Randomness...")
    key = jax.random.PRNGKey(0)
    key, rng, train_rng = jax.random.split(key, num=3)

    # Create dataset...
    trainloader, testloader, n_classes, seq_len = create_dataset_fn()

    print("[*] Starting Training =>> Initializing Model + Train State...")

    model = partial(
        BatchSeqModel, layer=model_cls, d_output=n_classes, d_model=64, n_layers=4, l_max=seq_len
    )
    state = create_train_state(model, rng, bsz=bsz, seq_len=seq_len)

    # Loop over epochs
    for epoch in range(epochs):

        print(f"[*] Running Epoch {epoch + 1} Validation...")
        test_loss, test_acc = validate(state.params, model, testloader)
        checkpoints.save_checkpoint("mnist", state, epoch)
        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        state, train_loss = train_epoch(state, train_rng, model, trainloader)

        print(
            f"\tTrain Loss: {train_loss:.5f} -- Test Loss: {test_loss:.5f} -- Test"
            f" Accuracy: {test_acc:.4f}\n"
        )



Models = {
    "ff": FeedForwardModel,
    "lstm": LSTMRecurrentModel,
}



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=Datasets.keys(), required=True)
    parser.add_argument("--model", type=str, choices=Models.keys(), required=True)
    args = parser.parse_args()

    example_train(Models[args.model], Datasets[args.dataset])
