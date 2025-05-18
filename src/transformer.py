"""Main modules for the transformer model."""

import jax
import jaxtyping as jt
from flax import nnx
import functools
import einops
import jax.numpy as jnp
import orbax.checkpoint as ocp
import src.my_types as my_types
import checkpoints
import os

Float = jt.Float
Int = jt.Int
Array = jt.Array


@functools.partial(jax.jit, static_argnames=("patch_size",))
def patchify_image(
    image: Float[Array, "B H W C"], patch_size: int
) -> Float[Array, "B (H W) (P P C)"]:
    return einops.rearrange(
        image,
        "... (h p1) (w p2) c -> ... (h w) (p1 p2 c)",  # h != Height, w != Width
        p1=patch_size,
        p2=patch_size,
    )


def split_attention_heads(x, num_heads, dim_head):
    # (B, N, D_qkv) -> (B, N, num_heads, D_head)
    batch_size, num_patches = x.shape[:2]
    x = x.reshape((batch_size, num_patches, num_heads, dim_head))  
    # (B, N, num_heads, D_head) -> (B, num_heads, N, D_head)
    x = jnp.transpose(x, (0, 2, 1, 3))  
    return x


def add_classification_token(
    cls_embedding: Float[Array, "1 D"], embeddings: Float[Array, "B N D"]
) -> Float[Array, "B M D"]:
    """Add classification token to the input."""

    cls_embedding = jnp.tile(
        cls_embedding, (embeddings.shape[0], 1, 1)
    )
    return jnp.concatenate([cls_embedding, embeddings], axis=1)


class PatchEmbedding(nnx.Module):
    """Patch embedding module."""

    def __init__(self, patch_size: int, out_features: int, rngs: nnx.Rngs):
        """Initialize the module.
        
        Args:
            patch_size: The size of the patch.
            out_features: The output dimension of the embedding.
            rngs: The random number generator state.
        """
        self.patch_size = patch_size
        self.out_features = out_features
        self.channels = 3
        self.linear = nnx.Linear(
            in_features=self.patch_size**2 * self.channels,
            out_features=self.out_features,
            rngs=rngs
        )

    def __call__(self, inputs: Float[Array, "B H W C"]) -> Float[Array, "B (H W) D"]:

        patches = patchify_image(inputs, self.patch_size)   # (B, H, W, C) -> (B, H*W, P*P*C)
        projection = self.linear(patches)   # (B, H*W, PPC) -> (B, H*W, D)

        return projection


class PositionalEmbedding(nnx.Module):

    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        """Initialize the Positional Embedding module.
        
        Args:
            in_features: The input dimension, typicaly number of patches (+ 1 for cls token).
            out_features: The output embedding dimension.
            rngs: The random number generator state.
        """
        self.in_features = in_features
        self.out_features = out_features
        key = rngs.params()
        self.pos_embedding = nnx.Param(
            jax.random.normal(key=key, shape=(self.in_features, self.out_features)) * 0.02
    )

    def __call__(self) -> Float[Array, "B N D"]:
        return self.pos_embedding


class LayerNorm(nnx.Module):
    """Layer normalization module."""

    def __call__(self, x: Float[Array, "B N D"]) -> Float[Array, "B N D"]:
        x_mean = jnp.mean(x, axis=-1, keepdims=True)
        x_std = jnp.std(x, axis=-1, keepdims=True)
        return (x - x_mean) / (x_std + 1e-6)


class MultiHeadAttention(nnx.Module):
    """Multi-head attention module."""

    def __init__(self, num_heads: int, dim_qkv: int, dim_emb: int, dim_out: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.dim_qkv = dim_qkv
        self.dim_emb = dim_emb
        self.dim_out = dim_out

        self.head_dim_qkv = self.dim_qkv // self.num_heads
        self.head_dim_emb = self.dim_emb // self.num_heads

        self.q = nnx.Linear(                    # (B N D_sa) -> (B N D_qk)
            in_features=self.dim_emb,
            out_features=self.dim_qkv,
            rngs=nnx.Rngs(params=rngs.params())
        )
        self.k = nnx.Linear(                    # (B N D_sa) -> (B N D_dk)
            in_features=self.dim_emb,
            out_features=self.dim_qkv,
            rngs=nnx.Rngs(params=rngs.params())
        )
        self.v = nnx.Linear(                    # (B N D_sa) -> (B N D_v)
            in_features=self.dim_emb,
            out_features=self.dim_qkv,
            rngs=nnx.Rngs(params=rngs.params())
        )
        self.out = nnx.Linear(                    # (B N D_v) -> (B N D_o)
            in_features=self.dim_qkv,
            out_features=self.dim_out   ,
            rngs=nnx.Rngs(params=rngs.params())
        )
        self.attention_dropout = nnx.Dropout(
            rate=0.1,
            rngs=nnx.Rngs(dropout=rngs.params())
        )

    def __call__(self, x: Float[Array, "B N D"]) -> Float[Array, "B N D"]:

        q_proj = self.q(x)  # (B, N, D_qk)
        k_proj = self.k(x)
        v_proj = self.v(x)

        qkv = jnp.stack([q_proj, k_proj, v_proj], axis=0) 
        
        # (B, N, D) -> (B, num_heads, N, D_head)
        qkv_split = jax.vmap(split_attention_heads, (0, None, None))(qkv, self.num_heads, self.head_dim_qkv)  
        q, k, v = qkv_split[0], qkv_split[1], qkv_split[2]
        
        scores = jnp.einsum('bhnd,bhmd->bhnm', q, k)
        scores = scores / jnp.sqrt(self.head_dim_qkv)
        
        attention_weights = nnx.softmax(scores, axis=-1)
        attention_heads = attention_weights @ v   # (B, num_heads, N, D_head)

        # (B, num_heads, N, D_head) -> (B, N, D)
        attention_heads = jnp.transpose(attention_heads, (0, 2, 1, 3))
        attention = attention_heads.reshape(
            (attention_heads.shape[0], attention_heads.shape[1], self.dim_qkv)
        )

        attention = self.out(attention)
        attention = self.attention_dropout(attention)

        return attention


class EncoderBlock(nnx.Module):
    """Encoder block for the transformer model."""

    def __init__(self, num_heads: int, dim_emb: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.dim_emb = dim_emb
        self.layer_norm = LayerNorm()
        self.rngs = rngs
        self.drop_rngs = nnx.Rngs(dropout=self.rngs.params())
        self.multi_head_attention = MultiHeadAttention(
            num_heads=self.num_heads,
            dim_emb=self.dim_emb,
            dim_qkv=self.dim_emb,
            dim_out=self.dim_emb,
            rngs=nnx.Rngs(params=self.rngs.params())
        )
        self.mlp = nnx.Sequential(
            nnx.Linear(
                in_features=self.dim_emb,
                out_features=self.dim_emb * 4,
                rngs=nnx.Rngs(params=self.rngs.params())
            ),
            nnx.gelu,
            nnx.Linear(
                in_features=self.dim_emb * 4,
                out_features=self.dim_emb,
                rngs=nnx.Rngs(params=self.rngs.params())
            )
        )
        self.drop = nnx.Dropout(
            rate=0.1,
            rngs=self.drop_rngs
        ) 

    def __call__(self, embeddings_in: Float[Array, "B N D"]) -> Float[Array, "B N D"]:

        x_norm1 = self.layer_norm(embeddings_in)
        x_attention = self.multi_head_attention(x_norm1)
        x_skip = x_attention + embeddings_in
        x_norm2 = self.layer_norm(x_skip)
        x_mlp = self.drop(self.mlp(x_norm2))
        # Dropout to prevent overfitting
        x_out = x_mlp + x_skip

        return x_out


class VisionTransformer(nnx.Module):
    """Transformer model."""

    def __init__(
        self,
        config: my_types.ConfigFile
    ):
        self.patch_size=config["patch_size"]
        self.dim_emb=config["dim_emb"]
        self.num_heads=config["num_heads"]
        self.num_encoder_blocks=config["num_encoder_blocks"]
        self.width=config["width"]
        self.height=config["height"]
        self.num_classes=config["num_classes"]
        
        self.num_patches = (self.width * self.height) // (self.patch_size**2)
        self.rngs = nnx.Rngs(42)

        self.patch_embedding_fn = PatchEmbedding(
            patch_size=self.patch_size,
            out_features=self.dim_emb,
            rngs=nnx.Rngs(params=self.rngs.params())
        )
        self.positional_embedding_fn = PositionalEmbedding(
            in_features=self.num_patches + 1,
            out_features=self.dim_emb,
            rngs=nnx.Rngs(params=self.rngs.params())
        )
        self.pos_embeddings = self.positional_embedding_fn()
        self.cls_embedding = nnx.Param(
            jax.random.normal(key=self.rngs.params(), shape=(1, self.dim_emb)) * 0.02
        )
        self.embedding_dropout = nnx.Dropout(
            rate=0.1,
            rngs=nnx.Rngs(dropout=self.rngs.params())
        )
        self.encoder_blocks = nnx.Sequential(*[
            EncoderBlock(num_heads=self.num_heads, dim_emb=self.dim_emb, rngs=nnx.Rngs(params=self.rngs.params()))
            for _ in range(self.num_encoder_blocks)
        ])
        self.linear = nnx.Linear(
            in_features=self.dim_emb,
            out_features=self.num_classes,
            rngs=nnx.Rngs(params=self.rngs.params())
        )

    def __call__(self, x: Float[Array, "B H W C"]) -> Float[Array, "B N D"]:
        """Forward pass."""

        x = x / 255.0 # Normalize pixel values

        patch_embeddings = self.patch_embedding_fn(x)
        embeddings = add_classification_token(
            self.cls_embedding, patch_embeddings
        )
        embeddings = embeddings + self.pos_embeddings
        embeddings = self.embedding_dropout(embeddings)

        vision_embeddings = self.encoder_blocks(embeddings)  # (B, num_patches+1, D)

        output = self.linear(vision_embeddings[:, 0, :])  # cls embedding -> (B, num_classes)

        return output, vision_embeddings
    
    def save_weights(self, path: str) -> None:
        _, state = nnx.split(self)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path, state)

    @classmethod
    def load_weights(cls, path: str, config: my_types.ConfigFile) -> None:
        """Boilerplate code to load weights from a checkpoint. """
        abstract_model = nnx.eval_shape(
            lambda: cls(config=config)
        )
        graphdef, abstract_state = nnx.split(abstract_model)

        checkpointer = ocp.PyTreeCheckpointer()
        checkpoints_dir = os.path.dirname(checkpoints.__file__)
        checkpoints_path = os.path.join(checkpoints_dir, path)
        state_restored = checkpointer.restore(checkpoints_path, abstract_state)
        
        # The model is now good to use!
        model = nnx.merge(graphdef, state_restored)
        return model
