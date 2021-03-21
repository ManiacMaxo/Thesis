import tensorflow as tf
from tensorflow.keras import layers
from einops import rearrange


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, dim: int, dim_qkv=16, heads=8, dropout=0.0):
        super().__init__()
        self.inner_dim = dim_qkv * heads
        self.heads = heads
        self.scale = dim_qkv ** -0.5

        self.to_qkv = layers.Dense(dim, self.inner_dim * 3, use_bias=False)
        self.to_out = layers.Dense(self.inner_dim, dim)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = tf.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = sim.softmax(-1)
        attn = self.dropout(attn)

        out = tf.einsum("b h n m, b h m d -> b h n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        return self.to_out(out)
