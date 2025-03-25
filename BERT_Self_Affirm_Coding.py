# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 23:01:18 2023

@author: cye
"""
# Using Python 3.9
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import keras

# SET WORKING FOLDER AND DATA TO CODE
folder = "C:/Users/cye/Documents"
excel_file = "essays.xlsx"

os.chdir(folder)
@keras.saving.register_keras_serializable()
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

@keras.saving.register_keras_serializable()
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

@keras.saving.register_keras_serializable()
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1'
bert_layer = hub.KerasLayer(module_url, trainable=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
model=tf.keras.models.load_model('model_H24A1024_epoch100.keras', custom_objects={'KerasLayer':hub.KerasLayer})

import tokenization
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=512):
    # bert can support max length of 512 only 
    # here we need 3 data inputs for bert training and fine tuning 
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2] # here we are trimming 2 words if they getting bigger than 512
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

data = pd.read_excel(excel_file)
data.dropna(subset=['Text'],inplace=True, how='all') #Text contains essays
test_input = bert_encode(data.Text.values, tokenizer, max_len=160)
test_pred = model.predict(test_input)
data['BERT_coded'] = np.where(test_pred>.5, 1,0)
print(data['BERT_coded'].value_counts())
