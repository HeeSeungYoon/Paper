import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Softmax
from konlpy.tag import Okt
from tqdm import tqdm
import re
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import time_log

def load_review_data(file_name, col_name):
    df = pd.read_csv(file_name, sep='\t', engine='python', encoding='utf-8')
    reviews = df[col_name].to_numpy()

    return reviews

def refine_text_data(text_data):
    
    okt = Okt()

    stopword_df = pd.read_csv('stopwords.csv',index_col=False)
    stopword = stopword_df['stopword'].to_numpy()

    # 1. remain only kor
    for i in tqdm(range(len(text_data)), desc='Remaining only korean'):
        try:
            text_data[i] = re.sub('[^가-힣]', ' ', text_data[i])
        except TypeError:
            text_data[i] = ''

    # 2. normalize and convert to morpheme
    refining_text_data = []
    for i in tqdm(range(len(text_data)), desc='Normalizing and morpheme analyzing'):
        if len(text_data[i]) > 0:        
            text_data[i] = okt.normalize(text_data[i])
            text_data[i] = okt.morphs(text_data[i])
            refining_text_data.append(text_data[i])

    refined_text_data = []
    # 3. remove too short review and stopword
    for morphs in tqdm(refining_text_data, desc='removing too short review and stopword'):
        text = []
        for morph in morphs:
            if len(morph) > 1 and morph not in stopword:
                text.append(morph)
        if len(text) > 0:
            refined_text_data.append(text)
    
    return refined_text_data

def one_hot_encoding(tokens, word_to_index):
    encoded_data = []
    
    
    for token in tokens:
        one_hot_vector = [0] * len(word_to_index)
        index = word_to_index[token]
        one_hot_vector[index] = 1
        encoded_data.append(one_hot_vector)

    return encoded_data


def preprocess_data(refined_text_data, N):

    word_to_index = {'':0}
    idx = 1
    for tokens in tqdm(refined_text_data, desc='making word dictionary'):
        for token in tokens:
            if token not in word_to_index:
                word_to_index[token] = idx
                idx += 1

    padding_data = []
    target = []
    for tokens in tqdm(refined_text_data, desc='padding and setting target'):
        if len(tokens) < N:
            padding = tokens + ['']*(N-len(tokens))
            padding_data.append(padding)
            target.append([''])
        elif len(tokens) > N:
            padding_data.append(tokens[:N])
            target.append([tokens[N]])
            for i in range(N,len(tokens)):
                padding_data.append(tokens[i-N+1:i+1])
                target.append([tokens[i+1]] if i+1 < len(tokens) else [''])
        else:
            padding_data.append(tokens)
            target.append([''])

    preprocessed_data = []
    for tokens in tqdm(padding_data, desc='one-hot encoding input data'):
        encoded_data = one_hot_encoding(tokens, word_to_index)
        preprocessed_data.append(encoded_data)

    encoded_target = []
    for word in tqdm(target, desc='one-hot encoding target'):
        encoded_data = one_hot_encoding(word, word_to_index)
        encoded_target.append(encoded_data)

    print(f'input data shape: {np.shape(preprocessed_data)}')
    print(f'target data shape: {np.shape(encoded_target)}')
    
    return [np.array(preprocessed_data, dtype=float), np.array(encoded_target,dtype=float)]

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
        

if __name__ == '__main__':

    start_time = time.time()
    # Job 1: load review data and split into train/test 
    path = os.getcwd()
    review = load_review_data(path+'/Word2Vec/nsmc.txt', col_name='document')

    train, test = train_test_split(review, test_size=0.25, shuffle=True)
    time_log('\nload data and split into train/test')

    # Job 2: refining data
    train = refine_text_data(train[:100])    
    time_log('refining train data')
    test = refine_text_data(test[:25])    
    time_log('refining test data')

    # Job 3: preprocessing data
    N = 10
    train, train_target = preprocess_data(train, N)
    time_log('preprocessing train data')
    test, test_target = preprocess_data(test, N)
    time_log('preprocessing test data')

    # Job 4: NNLM Model design
    B, _, V = train.shape
    test_B, _, test_V = test.shape
    train_target = np.reshape(train_target, (B, V))
    test_target = np.reshape(test_target, (test_B, test_V))

    nnlm = NNLM(N, V, 1000, 500)
    y = nnlm(train)
    nnlm.summary()

    nnlm.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = nnlm.fit(train, train_target, epochs=10)

    time_log('total execute', start_time=start_time)
    
    