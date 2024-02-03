import tensorflow as tf
from tensorflow.keras import Model

from nnlm import ProjectionLayer

class CBOW(Model):
    def __init__(self, N, V, D):
        super(CBOW, self).__init__()

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(N, V))

        self.embedding_layer = ProjectionLayer(D)

        self.output_layer = tf.keras.layers.Dense(V)

        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        
        x = self.input_layer(inputs)

        x = self.embedding_layer(x)
        # average N word vectors
        x = tf.reduce_mean(x, 1)

        x = self.output_layer(x)

        return self.softmax(x)