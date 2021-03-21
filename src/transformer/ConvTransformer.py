import tensorflow as tf

from . import TransformerBlock


class ConvTransformer(tf.keras.Model):
    """ Transcription Factor locator

    """

    def __init__(self, embed_dim, num_heads, ff_dim, num_labels=1):
        """Convolutional transformer based on DanQ

        Args:
            embed_dim ([type]): Embedding size for each token
            num_heads ([type]): Number of attention heads
            ff_dim ([type]): Hidden layer size in feed forward network inside transformer
            num_labels (int, optional): Number of outputs. Defaults to 1.
        """
        super().__init__()
        self.input = tf.keras.layers.Input(shape=(1000, 4))
        self.conv1d = tf.keras.layers.Conv1D(
            embed_dim, kernel_size=26, activation="relu"
        )
        self.max_pooling = tf.keras.layers.MaxPooling1D(pool_size=13, strides=13)
        self.dropout_1 = tf.keras.layers.Dropout(0.2)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = Dense(925, activation="relu")
        self.outputs = Dense(num_labels)

    def call(self, inputs, training=False):
        x = self.input(inputs)
        x = self.conv1d(x)
        x = self.max_pooling(x)
        if training:
            x = self.dropout_1(x)
        x = self.transformer(x)
        if training:
            x = self.dropout_2(x)
        x = self.dense(x)
        return self.outputs(x)
