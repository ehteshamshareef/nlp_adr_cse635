
# coding: utf-8

# In[ ]:





# In[1]:



import tensorflow as tf
import sys
import itertools
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertForTokenClassification


# In[2]:



# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Load the BERT tokenizer.
# print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('Task1/biobert_task1', do_lower_case=False)


# In[3]:




# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "Task1/biobert_task1", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)


# In[ ]:





# In[4]:


def make_predictions(sentence,model = model,tokenizer = tokenizer):
    input_ids = []
    attention_masks = []
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    encoded_dict = tokenizer.encode_plus(
                            sentence,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 64,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # labels = torch.tensor(labels)

    # Set the batch size.  
    batch_size = 32  

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    predictions = []
    for batch in prediction_dataloader:
      # Add batch to GPU
    #   batch = tuple(t.to(device) for t in batch)

      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask = batch

      # Telling the model not to compute or store gradients, saving memory and 
      # speeding up prediction
      with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)

      logits = outputs[0]

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()

      # Store predictions and true labels
      predictions.append(logits)
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    if flat_predictions[0] == 0:
        return "Not ADR"
    else:
        return "ADR"


# In[5]:


make_predictions("my new medication made me feel the most depressed ever")


# In[6]:


#Task2


# In[7]:


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences


# In[8]:


tokenizer = BertTokenizer.from_pretrained('task2_bert', do_lower_case=False)


# In[9]:


import transformers
from transformers import BertForTokenClassification, AdamW


# In[10]:


model_task2 = BertForTokenClassification.from_pretrained(
    "task2_bert",
    num_labels=4,
    output_attentions = False,
    output_hidden_states = False
)


# In[42]:


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
#         print("{}\t{}".format(label, token))
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
            
            


# In[43]:


get_extraction("all this cipro is making my so sleepy")


# In[13]:


import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns


# In[14]:


module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = hub.load(module_url)
def embed(input):
  return model(input)


# In[15]:


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


# In[16]:


def get_normalized(input_array,actual_embeddings):
    extracted_embeddings = embed(input_array)
    cosine = cosine_similarity(extracted_embeddings, actual_embeddings)
    indexes = list(np.argmax(cosine, axis=1))
    predicted = []
    for item in indexes:
      predicted.append(unique_actual[item])
    return predicted


# In[17]:


get_normalized(["depressed"],actual_embeddings)


# In[51]:


phrase = "all this cipro is making me unable to sleep"
isADR = make_predictions(phrase)
if (isADR):
    extractions = get_extraction(phrase)
    meddra = get_normalized(extractions,actual_embeddings)
    print (extractions,meddra) 


# In[18]:




