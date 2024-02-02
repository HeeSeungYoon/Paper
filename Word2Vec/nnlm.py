import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Softmax

class ProjectionLayer(tf.keras.layers.Layer):
    def __init__(self, D):
        super(ProjectionLayer, self).__init__()
        self.D = D
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.D), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.D, ), initializer='zero', trainable=True)
    
    def call(self, inputs):
        # Embedding Vector
        return tf.matmul(inputs, self.w) + self.b

class NNLM(Model):
    def __init__(self, N, V, D, H):
        super(NNLM, self).__init__()
        self.input_layer = InputLayer(input_shape=(N, V))
        self.projection_layer = ProjectionLayer(D)
        self.flatten = Flatten()
        self.hidden_layer = Dense(H)
        self.output_layer = Dense(V)
        self.softmax = Softmax()
    
    def call(self, inputs):
        x = self.input_layer(inputs)

        x = self.projection_layer(x)        
        x = self.flatten(x)

        x = self.hidden_layer(x)

        x = self.output_layer(x)

        return self.softmax(x)