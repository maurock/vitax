from datasets import load_dataset, Dataset, load_from_disk
import src.my_types as my_types
from src.utils import make_dir
import yaml
import os
from typing import Dict, Any, Tuple
import data
import PIL
import data
import jax.numpy as jnp


def make_jax_dataset(
    dataset: Dataset,
    batch_size: int,
    drop_remainder: bool = True,
) -> my_types.MyData:
    """Converts a HuggingFace dataset into a simple JAX dataset.
    
    This allows to vmap over the dataset and use it in JAX training loops."""
    image_batches = []
    label_batches = []
    for a in dataset.iter(batch_size=batch_size, drop_last_batch=drop_remainder):
        image_batches.append(a['image'])
        label_batches.append(a['label'])    

    return my_types.MyData(
        image=jnp.stack(jnp.array(image_batches), axis=0),
        label=jnp.stack(jnp.array(label_batches), axis=0),
    )


def resize_fn(
    examples: Dict[str, Any], target_shape: Tuple[int, int, int]
) -> Dict[str, Any]:
    """
    Resizes the image to the target shape and normalizes pixel values.

    Args:
        examples: A dictionary containing 'image; and 'label'.
        target_shape: Desired shape (height, width, channels) for the image.

    Returns:
        Updated dictionary with resized and normalized image.
    """

    height, width = target_shape[:2]
    
    examples['image'] = examples['image'].resize(
        (width, height),
        PIL.Image.LANCZOS
    )

    return examples


class HF_Dataset(Dataset):
    """A wrapper for creating and saving datasets from HuggingFace sources."""

    def __init__(
        self,
        config: my_types.ConfigFile,
        from_disk: bool = False,
    ) -> None:
        self.config = config
        self.from_disk = from_disk

    def make_dataset(self) -> dict[str, Dataset]:
        """Loads and processes the dataset splits."""

        output_dataset = {}

        if self.from_disk:
            for split in self.config["splits"]:
                try:
                    data_dir = os.path.dirname(data.__file__)
                    path = os.path.join(data_dir, self.config["path"], split)
                    dataset = load_from_disk(path)
                    dataset = dataset.with_format("jax")
                    output_dataset[split] = dataset
                except Exception as e:
                    print(f"Error loading split {split}: {e}.")
                    continue
        else:
            for split in self.config["splits"]:
                output_dataset[split] = load_dataset(self.config["path"], split=split)
                output_dataset[split] = output_dataset[split].map(
                    resize_fn,
                    fn_kwargs={
                        "target_shape": (self.config["height"], self.config["width"], 3)
                    },
                )

        return output_dataset

    def save_dataset(self, my_dataset: Dict[str, Dataset], output_dir: str):
        """Save the dataset to disk.

        Args:
            data: The dataset to save.
            output_dir: The directory to save the dataset to. 
                Do not include the split name, as this will be added automatically.
        """

        for split in self.config["splits"]:
            path = os.path.join(output_dir, split)
            my_dataset[split].save_to_disk(path)
            print(f"Saved {split} dataset to {path}")

        path = os.path.join(output_dir, "config.yaml")
        with open(path, "w") as f:
            yaml.dump(self.config, f)


if __name__ == "__main__":
    config = my_types.ConfigFile(
        dict(
            path="Matthijs/snacks",
            width=160,
            height=160,
            splits=["train", "validation", "test"],
        )
    )

    hf_dataset = HF_Dataset(config)
    my_dataset = hf_dataset.make_dataset()

    output_dir = os.path.join(
        os.path.dirname(data.__file__), "snack_dataset.hf"
    )

    make_dir(output_dir)
    hf_dataset.save_dataset(my_dataset, output_dir)
