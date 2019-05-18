import torch
import torch.nn as nn 
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler

import time 
import json 
import fastText


class Model(nn.Module):
    def __init__(self, batch_size = 50, lr = 0.001, keep_prob = 0.4, class_num = 3, is_training = True):
        super(Model, self).__init__()

        self.batch_size = batch_size
        self.lr = lr 
        self.keep_prob = keep_prob
        self.class_num = class_num
        self.is_training = is_training

        self.conv1 = nn.Conv2d(2, 32, (5,5), stride = (2,2), padding = (2,2))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, (3,3), stride = (2,2), padding = (1,1))
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*50*25, 128)
        self.fc2 = nn.Linear(128, 64)
        self.logits = nn.Linear(64, self.class_num)

        self.layer_output = {}

    def forward(self, input):
        x = F.avg_pool2d(self.batch_norm1(F.relu(self.conv1(input))))
        x = F.avg_pool2d(self.batch_norm2(F.relu(self.conv2(x))))

        x = F.dropout2d(F.relu(self.fc1(x.view(-1,128*50*25))), p = self.keep_prob)
        x = F.dropout2d(F.relu(self.fc2(x)), p = self.keep_prob)
        x = self.logits(x)

        return x

class Trainer:
    def __init__(self, input_dir, model, epoch = 10, class_num = 3):
        self.input_dir = input_dir 
        self.model = model 
        self.epoch = epoch
        self.class_num = class_num

        self.lr = self.model.lr
        self.batch_size = self.model.batch_size
        

        self.loss = self.model.loss()
        self.accuracy = self.model.accuracy()
        self.optimizer = self.model.optimizer()

        self.caption_train = json.load(open(self.input_dir+"train_caption_json.json", 'r'))[:-300]
        self.caption_val = json.load(open(self.input_dir+"train_caption_json.json", 'r'))[-300:]
        self.reference_train = json.load(open(self.input_dir+"train_reference_json.json", 'r'))[:-300]
        self.reference_val = json.load(open(self.input_dir+"train_reference_json.json", 'r'))[-300:]
        self.negative_train = json.load(open(self.input_dir+"train_negative_json.json", 'r'))[:-300]
        self.negative_val = json.load(open(self.input_dir+"train_negative_json.json", 'r'))[-300:]

        # self.word2vec_model = word2vec.load('./embedding/word2vec.bin')
        self.word2vec_model = fastText.FastText.load_model('./embedding/fil9.bin')
    





def sentence_embedding(dic_list, word2vec_model, category):
    batch_list = []
    for dic in dic_list:
        token_list = nltk.word_tokenize(dic["Text"])
        
        sample = []

        sent_matrix = []
        pos_matrix = []

        
        for token in token_list:
            # word_vec = word2vec_model[token].tolist()
            word_vec = word2vec_model.get_word_vector(token).tolist()
            
            sent_matrix.append(word_vec)
            
            if token == ",":
                pos_matrix.append([1]*50+[0]*50)
            elif token == ".":
                pos_matrix.append([0]*50+[1]*50)
            else:
                # pos_matrix.append(word2vec_model[str(position)])
                pos_matrix.append(word2vec_model.get_word_vector(token))
        
        sent_matrix += [[0]*100]*(200-len(sent_matrix))
        pos_matrix += [[0]*100]*(200-len(pos_matrix))
        sent_matrix = np.expand_dims(np.array(sent_matrix), axis = -1)
        pos_matrix = np.expand_dims(np.array(pos_matrix), axis = -1)
        concated_matrix = np.concatenate((sent_matrix, pos_matrix), axis = -1).tolist()

        sample.append([concated_matrix, category])

        batch_list.append(sample)
    
    return batch_list
