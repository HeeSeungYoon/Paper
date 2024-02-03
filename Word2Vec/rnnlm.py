import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Softmax

from nnlm import ProjectionLayer

class HiddenLayer(tf.keras.layers.Layer):
    def __init__(self, H):
        super(HiddenLayer, self).__init__()
        self.H = H
        
    def build(self, input_shape):
        self.wx = self.add_weight(shape=(input_shape[-1], self.H), initializer='random_normal', trainable=True)
        self.wh = self.add_weight(shape=(self.H, self.H), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(1, self.H), initializer='zero', trainable=True)
        self.ht_1 = self.add_weight(shape=(1, self.H), initializer='zero', trainable=False)
    
    def call(self, inputs):
        
        batch, time_step, _ = inputs.shape

        # M input, 1 output
        for t in range((time_step)):
            word = inputs[:,t,:]
            word = tf.reshape(word, [-1, 1, self.H])
            ht_1 = tf.matmul(self.ht_1, self.wh)
            x = tf.matmul(word, self.wx)
            ht = ht_1 + x + self.b
            ht = tf.keras.activations.tanh(ht)
            self.ht_1 = ht
        return ht

class RNNLM(Model):
    def __init__(self, N, V, H):
        super(RNNLM, self).__init__()
        self.input_layer = InputLayer(input_shape=(N,V))
        self.embedding_layer = ProjectionLayer(H)
        # self.hidden_layer = HiddenLayer(H)
        self.hidden_layer = tf.keras.layers.SimpleRNN(N, input_shape=(N, H))
        self.output_layer = Dense(V)
        self.softmax = Softmax()

    def call(self, inputs):

        x = self.input_layer(inputs)
        
        # Embedding
        x = self.embedding_layer(x)

        # RNN
        x = self.hidden_layer(x)
        
        x = self.output_layer(x)

        return self.softmax(x)