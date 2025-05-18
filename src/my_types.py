"""Define custom types"""

import ml_collections
import typing
import chex
import jaxtyping as jt
import jax

Float = jt.Float
Array = jt.Array
Int = jt.Int

ConfigFile = typing.NewType("ConfigFile", ml_collections.ConfigDict)


@chex.dataclass
class DataPoint:
    """A simple JAX container for a data point."""
    image: Float[Array, "H W C"]
    label: Int[Array, ""]


@chex.dataclass
class MyData:
    """A simple JAX container for my dataset."""
    image: Float[Array, 'N B H W C']
    label: Int[Array, 'N B']

    def sample_by_index(
        self,
        index: int,
    ) -> "DataPoint":
        """Sample the dataset by index."""
        image = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]),
            self.image
        )[index]
        label = jax.tree_util.tree_map(
            lambda x: x.reshape(-1),
            self.label
        )[index]

        return DataPoint(
            image=image,
            label=label,
        )
