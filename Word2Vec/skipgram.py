import tensorflow as tf
from tensorflow.keras import Model

from nnlm import ProjectionLayer

class SkipGram(Model):
    def __init__(self, N, V, D):
        super(SkipGram, self).__init__()

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(V))

        self.embedding_layer = ProjectionLayer(D)

        self.output_layer = tf.keras.layers.Dense(N*V)

        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        
        x = self.input_layer(inputs)

        x = self.embedding_layer(x)

        x = self.output_layer(x)

        return self.softmax(x)