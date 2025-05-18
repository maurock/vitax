"""Functions to patchify images."""

import jax
import chex
import jaxtyping as jt
import src.my_types as my_types
import einops
import functools

Float = jt.Float
Array = jt.Array


@chex.dataclass(frozen=True)
class Patchifier:
    """Patchify images"""

    patch_size: int | None = None

    @staticmethod
    def make_patchifier(
        config: my_types.ConfigFile,
    ) -> "Patchifier":
        """Create a patchify object."""
        return Patchifier(
            patch_size=config["patch_size"],
        )

    @functools.partial(jax.jit, static_argnames=("patch_size",))
    def _patchify_image(self, image, patch_size):
        return einops.rearrange(
            image,
            "... (h p1) (w p2) c -> ... (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        )

    def patchify_image(
        self, image: Float[Array, "*B H W C"]
    ) -> Float[Array, "*B (H W) (P P C)"]:
        """Patchify a batch of images."""
        return self._patchify_image(image, patch_size=self.patch_size)
