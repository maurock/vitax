"""Train the model."""

import optax
import jaxtyping as jt
from src import transformer, my_types, dataset
import yaml
from flax import nnx
import jax.numpy as jnp
import jax
from datasets import Dataset
from typing import Tuple
from tqdm import tqdm
import configs
import os
import checkpoints

Float = jt.Float
Array = jt.Array
Int = jt.Int


def cross_entropy(
    prediction: Int[Array, "B C"], label: Int[Array, "B C"]
) -> Float[Array, "B"]:
    """Compute the cross-entropy loss."""

    log_probs = nnx.log_softmax(prediction, axis=-1)
    cross_entropy = -jnp.sum(label * log_probs, axis=-1)
    mean_cross_entropy = jnp.mean(cross_entropy)

    return mean_cross_entropy


def loss_fn(
    model: transformer.VisionTransformer, batch: dict[str, Array]
) -> Float[Array, "B"]:
    """Compute the loss."""

    prediction, _ = model(batch["image"])
    label = batch["label"]
    label = jax.nn.one_hot(label, num_classes=model.num_classes)
    loss = cross_entropy(prediction, label)

    return loss


@nnx.jit
def train_step(
    model: transformer.VisionTransformer,
    optimizer: nnx.Optimizer,
    batch: dict[str, Array],
) -> Tuple[transformer.VisionTransformer, nnx.Optimizer]:
    """Perform a single training step."""

    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(grads)

    return model, optimizer, loss


def run_evaluation(
    model: transformer.VisionTransformer,
    validation_data: dict[str, Array]
) -> Float[Array, "B"]:
    """Run evaluation"""

    @nnx.scan
    def run_evaluation_step(carry_model: nnx.Carry, batch: my_types.MyData) -> Array:
        """Run a single evaluation step."""
        loss_value = loss_fn(carry_model, batch)
        return carry_model, loss_value

    _, mean_val_loss = run_evaluation_step(model, validation_data)
    mean_val_loss = jnp.mean(mean_val_loss)

    return mean_val_loss


def run_training(
    config: my_types.ConfigFile,
    model: transformer.VisionTransformer,
    train_data: my_types.MyData,
    validation_data: my_types.MyData = None,
) -> transformer.VisionTransformer:
    """Run full training."""

    @nnx.scan
    def train_single_batch(
        carry: nnx.Carry,
        batch: my_types.MyData,
    ) -> Tuple[nnx.Carry, Float[Array, ""]]:
        """Processes a single batch within an epoch.

        With nnx.scan, the model and optimizer are updated and brought forward.
        """
        model, optimizer = carry
        model, optimizer, loss_value = train_step(model, optimizer, batch)

        return (model, optimizer), loss_value

    # Initialize the optimizer
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=config["learning_rate"]))

    for epoch_idx in range(config["num_epochs"]):
        model.train()  # Sets `deterministic=False` for nnx.Dropout
        (model, optimizer), all_batch_losses = train_single_batch((model, optimizer), train_data)
        mean_epoch_loss = jnp.mean(all_batch_losses)
        print(
            f"Epoch {epoch_idx + 1}/{config['num_epochs']}, Mean Loss: {mean_epoch_loss:.4f}"
        )

        if validation_data is not None:
            model.eval()
            mean_val_loss = run_evaluation(model, validation_data)
            print(f"Validation Loss: {mean_val_loss:.4f}")

    return model  # Return final model


if __name__ == "__main__":
    # Load the configuration file
    configs_dir = os.path.dirname(configs.__file__)
    config_file = yaml.safe_load(open(os.path.join(configs_dir, "config.yml"), "r"))
    config = my_types.ConfigFile(config_file)

    model = transformer.VisionTransformer(
        config=config
    )

    # Load the Hugging Face dataset
    hf_dataset = dataset.HF_Dataset(config, from_disk=True)
    my_dataset = hf_dataset.make_dataset()
    train_ds = my_dataset["train"].shuffle()
    val_ds = my_dataset["validation"].shuffle()
    test_ds = my_dataset["test"].shuffle()

    # Create a JAX dataset so we can scan over it
    my_train_data = dataset.make_jax_dataset(
        train_ds, batch_size=config["batch_size"], drop_remainder=True
    )
    # Optionally we can also create a validation dataset
    my_val_data = dataset.make_jax_dataset(
        val_ds, batch_size=config["batch_size"], drop_remainder=True
    )

    model_optimized = run_training(
        config=config,
        model=model,
        train_data=my_train_data,
        validation_data=my_val_data,
    )

    # Save the model
    output_dir = os.path.dirname(checkpoints.__file__)
    output_path = os.path.join(output_dir, config["output_name"])
    model.save_weights(output_path)
    print(f"Model saved to {output_path}")
    print("Training complete.")
