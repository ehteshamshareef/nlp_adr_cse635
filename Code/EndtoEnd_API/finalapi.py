import os
import json
import urllib.request
import pymongo
from  pymongo import MongoClient
from flask import Flask,request,jsonify
from flask_cors import CORS	
from datetime import datetime, timedelta
import tensorflow as tf
import sys
import itertools
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import torch
from torch import nn
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertForTokenClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
import transformers
from transformers import BertForTokenClassification, AdamW

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# cors = CORS(app, resources={r"*": {"origins": ""}})
CORS(app, resources=r'*', headers='Content-Type')
# client = MongoClient('mongodb://127.0.0.1:27017')
# db = client["testdb"]
# collection = db["tweets"]

tokenizer = BertTokenizer.from_pretrained('Task1/biobert_task1', do_lower_case=False)


# In[3]:





model = BertForSequenceClassification.from_pretrained(
    "Task1/biobert_task1",
    num_labels = 2,          
    output_attentions = False, 
    output_hidden_states = False, 
)


def make_predictions(sentence,model = model,tokenizer = tokenizer):
    input_ids = []
    attention_masks = []
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    encoded_dict = tokenizer.encode_plus(
                            sentence,                     
                            add_special_tokens = True, 
                            max_length = 64,           
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',    
                       )
      
    input_ids.append(encoded_dict['input_ids'])

    attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)



    batch_size = 32  


    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    predictions = []
    for batch in prediction_dataloader:

      b_input_ids, b_input_mask = batch


      with torch.no_grad():

          outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)

      logits = outputs[0]


      logits = logits.detach().cpu().numpy()

      predictions.append(logits)
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    if flat_predictions[0] == 0:
        return "Not ADR"
    else:
        return "ADR"



tokenizer = BertTokenizer.from_pretrained('task2_bert', do_lower_case=False)


model_task2 = BertForTokenClassification.from_pretrained(
    "task2_bert",
    num_labels=4,
    output_attentions = False,
    output_hidden_states = False
)

def get_extraction(test_sentence,model = model_task2):
    tag_values_test = ['I-ADR', 'O', 'B-ADR', 'PAD']
    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence])

    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values_test[label_idx])
            new_tokens.append(token)

    entities = []
    current = []
    for token, label in zip(new_tokens, new_labels):
        if label != "O" and token!='[CLS]' and token!="[SEP]":
            if label == "B-ADR" and len(current)!=0:
                entities.append(" ".join(current))
                current = []
                current.append(token)
            else:
                current.append(token)
        else:
            continue
    if len(current)!=0:
        entities.append(" ".join(current))
    return entities
            



module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = hub.load(module_url)
def embed(input):
  return model(input)


data_test = pd.read_csv("task3_validation (1).tsv", delimiter = "\t", encoding="utf-8-sig").fillna(method="ffill")
data_train = pd.read_csv("task3_training (2).tsv" ,delimiter = "\t", encoding="utf-8-sig").fillna(method="ffill")
data = pd.concat([data_train,data_test])
data= data.loc[:, ~data.columns.str.contains('^Unnamed')]
data = data.dropna(subset=['type'])
actual_terms = []
for index,row in data.iterrows():
  actual_terms.append(row.meddra_term)
unique_actual = np.unique(actual_terms)
actual_embeddings = torch.load('file.pt')





def get_normalized(input_array,actual_embeddings):
    extracted_embeddings = embed(input_array)
    cosine = cosine_similarity(extracted_embeddings, actual_embeddings)
    indexes = list(np.argmax(cosine, axis=1))
    predicted = []
    for item in indexes:
      predicted.append(unique_actual[item])
    return predicted






@app.route('/getTweets')
def get():
    phrase = request.args.get('phrase')
    isADR = make_predictions(phrase)
    results = []
    if (isADR == "ADR"):
        extractions = get_extraction(phrase)
        if extractions:
            meddra = get_normalized(extractions,actual_embeddings)
            for i in range(len(extractions)):
                result = {
                    "extraction" : "",
                    "meddra" : ""
                }
                result["extraction"] = extractions[i]
                result["meddra"] = meddra[i]
                results.append(result)
    return jsonify({"data" : results})


if __name__ == "__main__":
    app.run(host = "0.0.0.0",port = 80)