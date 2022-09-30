# Author: Xinwen Xu
# Date: 2022/8/1

from asyncore import read
import imp
import os
import pandas as pd
import torch
import  torch.nn as nn
from sklearn.model_selection import train_test_split

'''
data info:
ID: 424
data: 67453
after 2nd process, num(data) to model is 67335
'''

def create_char_set(data):
    words = set()
    strokes = set()
    for i,row in data.iterrows():
        burst = str(row["burst"])
        words.update(burst.split())
        raw = str(row["raw_burst"])
        strokes.update(list(raw))
    return words, strokes


# creat dictionaries for words and strokes
class create_char_dic():
    def __init__(self, chars):
        self.dict_char_id = dict()
        self.dict_id_char = list()
        
        for idx, char in enumerate(chars):
            self.dict_char_id[char] = idx
            self.dict_id_char.append(char)
    
    def char_to_id(self, char):
        return self.dict_char_id[char]
    
    def id_to_char(self, idx):
        return self.dict_id_char[idx]


# transfer to tensor
def sequence_to_tensor(word_dict, sequence, device, burst:bool):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if burst:
        seq = [word_dict.char_to_id(word) for word in str(sequence).split()]
    else:
        seq = [word_dict.char_to_id(word) for word in str(sequence)]
    return torch.tensor(seq).to(device)

def data_to_tensor(word_dict, data, device, burst=True):
    return [sequence_to_tensor(word_dict, sequence, device, burst) for sequence in data]
 

data = pd.read_excel("data.xls")
bursts = data["burst"]
raw = data["raw_burst"]
labels = data["pct_pause"]    
IDs = data["ID"]
types = data["type"]

words,strokes = create_char_set(data)       #(len(words),len(strokes))  --> (38848,161)

words_dict = create_char_dic(words)
strokes_dict = create_char_dic(strokes)     #len=67453

# split into train dataset and test dataset
def load(device, partition=0.8):  
    bursts_tensor = data_to_tensor(words_dict, bursts, device, burst=True)
    strokes_tensor = data_to_tensor(strokes_dict, raw, device, burst=False)

    bursts_train, bursts_test, \
    strokes_train, strokes_test, \
    IDs_train, IDs_test, \
    labels_train, labels_test = \
        train_test_split(bursts_tensor, strokes_tensor, IDs, labels, train_size=partition, random_state=42)

    train_data = list(zip(bursts_train, strokes_train, IDs_train, labels_train))
    test_data = list(zip(bursts_test, strokes_test, IDs_test, labels_test))
    
    return train_data, test_data
    
def load_entire(device):
    bursts_tensor = data_to_tensor(words_dict, bursts, device, burst=True)
    strokes_tensor = data_to_tensor(strokes_dict, raw, device, burst=False)

    #data = list(zip(bursts_tensor, strokes_tensor, IDs, types, labels))
    data = pd.DataFrame({"burst":bursts_tensor, "strokes":strokes_tensor, "ID":IDs, "type":types, "label":labels})

    return data

def info():
    return len(data), len(words), len(strokes), len(set(IDs)), len(types)


'''
Convert one piece of data to tensor, for getting a specified representation
input: the index of burst, which can be found in "data.xls"
'''
def convert_to_tensor(burst, strokes, device):
   # burst = data.iloc[b_index]["burst"]
   # raw_b = data.iloc[b_index]["raw_burst"]
    b_tensor = sequence_to_tensor(words_dict, burst, device, burst=True)
    s_tensor = sequence_to_tensor(strokes_dict, strokes, device, burst=False)
    return (b_tensor, s_tensor)

if __name__ == "dataloader":
    if not os.path.exists("data.xls"):
        os.system("dataprocessing.py")