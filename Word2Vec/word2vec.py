import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from konlpy.tag import Okt
from tqdm import tqdm
import re
import time
import platform

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import time_log
from nnlm import NNLM
from rnnlm import RNNLM

def load_review_data(file_name, col_name):
    df = pd.read_csv(file_name, sep='\t', engine='python', encoding='utf-8')
    reviews = df[col_name].to_numpy()

    return reviews

def refine_text_data(text_data):
    
    okt = Okt()

    stopword_df = pd.read_csv('stopwords.csv',index_col=False)
    stopword = stopword_df['stopword'].to_numpy()

    # 1. remain only kor
    kor_text = []
    for text in tqdm(text_data, desc='Remaining only korean'):
        try:
            kor = re.sub('[^가-힣]', ' ', text)
        except TypeError:
            kor = ''
        kor_text.append(kor)

    # 2. normalize and convert to morpheme
    refining_text_data = []
    for text in tqdm(kor_text, desc='Normalizing and morpheme analyzing'):
        if len(text) > 0:        
            morphs = okt.normalize(text)
            morphs = okt.morphs(text)
            refining_text_data.append(morphs)

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

def make_word_directory(text_data, word_to_index):
    idx = len(word_to_index)
    
    for tokens in tqdm(text_data, desc='making word dictionary'):
        for token in tokens:
            if token not in word_to_index:
                word_to_index[token] = idx
                idx += 1

def one_hot_encoding(tokens, word_to_index):
    encoded_data = []
    
    for token in tokens:
        try:
            one_hot_vector = [0] * len(word_to_index)
            index = word_to_index[token]
        except KeyError:
            index = 0
            pass
        
        one_hot_vector[index] = 1
        encoded_data.append(one_hot_vector)

    return encoded_data

def preprocess_data(refined_text_data, word_to_index, N, log=''):

    N_data = []
    target = []
    for tokens in tqdm(refined_text_data, desc='reshape N words and setting target'):
        if len(tokens) < N:
            target.append([tokens[-1]])
            padding = tokens + ['']*(N-len(tokens))
            N_data.append(padding)
        elif len(tokens) > N:
            N_data.append(tokens[:N])
            target.append([tokens[N]])
            for i in range(N,len(tokens)):
                N_data.append(tokens[i-N+1:i+1])
                target.append([tokens[i+1]] if i+1 < len(tokens) else [tokens[-1]])
        else:
            N_data.append(tokens)
            target.append([tokens[-1]])

    preprocessed_data = []
    for tokens in tqdm(N_data, desc=f'one-hot encoding {log} data'):
        encoded_data = one_hot_encoding(tokens, word_to_index)
        preprocessed_data.append(encoded_data)

    preprocessed_target = []
    for tokens in tqdm(target, desc=f'one-hot encoding {log} target'):
        encoded_target = one_hot_encoding(tokens, word_to_index)
        preprocessed_target.append(encoded_target)

    print(f'{log} data shape: {np.shape(preprocessed_data)}')
    print(f'{log} target data shape: {np.shape(preprocessed_target)}')

    return [np.array(preprocessed_data, dtype=float), np.array(preprocessed_target, dtype=float)]

def predict_data(sample, model, word_to_index, N=10):

    _sample = refine_text_data(sample)
    # for text in sample:
    #     tokens = list(text.split())
    #     _sample.append(tokens)

    preporcessed_sample, _ = preprocess_data(_sample, word_to_index, N, log='sample')

    print()
    predictions = model.predict(preporcessed_sample)
    print()

    words = list(word_to_index)
    for i in range(len(predictions)):
        idx = np.argmax(predictions[i])
        print(f'{sample[i]} -> {words[idx]}')

if __name__ == '__main__':

    start_time = time.time()
    # Job 1: load review data and split into train/test 
    path = os.getcwd()

    if platform.system() == 'Windows':
        path += '\\Word2Vec\\nsmc.txt'
    elif platform.system() == 'Linux':
        path += '/Word2Vec/nsmc.txt'
    
    review = load_review_data(path, col_name='document')
    max_len = len(max(review, key=lambda x: len(str(x).split())))

    train, test = train_test_split(review, test_size=0.25, shuffle=True)
    time_log('load data and split into train/test')

    # Job 2: refining data
    train = refine_text_data(train[:100])    
    time_log('refining train data')
    test = refine_text_data(test[:25])    
    time_log('refining test data')

    # Job 3: preprocessing data
    N = 10
    word_to_index = {'':0}

    make_word_directory(train, word_to_index)
    make_word_directory(test, word_to_index)
    print(f'Total Word: {len(word_to_index)}')
    time_log('making word directory')

    train, train_target = preprocess_data(train, word_to_index, N, log='train')
    time_log('preprocessing train data')
    test, test_target = preprocess_data(test, word_to_index, N, log='test')
    time_log('preprocessing test data')

    # Job 4: NNLM Model design
    # B: total Batch size, V: Vocabulary size
    B, _, V = train.shape
    test_B, _, test_V = test.shape
    train_target = np.reshape(train_target, (B, 1, V))
    test_target = np.reshape(test_target, (test_B, 1, V))

    # nnlm = NNLM(N, V, 1000, 500)
    # y = nnlm(train)
    # nnlm.summary()

    rnnlm = RNNLM(N, V, 500)
    y = rnnlm(train)
    rnnlm.summary()

    # rnnlm.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
    # history = rnnlm.fit(train, train_target, epochs=10, batch_size=32, callbacks=[tensorboard_callback])
    # time_log('Model training')

    # Job 5: prediction
    
    # evaluation = rnnlm.evaluate(test, test_target)
    # print(f'test loss: {evaluation[0]}')
    # print(f'test accuracy: {evaluation[1]}\n')

    # sample = ['호러 액션 스펙타클 재미와 감동의 쓰나미']
    # predict_data(sample, rnnlm, word_to_index)
    # time_log('Prediction')

    time_log('total execute', start_time=start_time)
    
    