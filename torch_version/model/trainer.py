import torch
import torch.nn as nn 
import torch.nn.functional as F

import torch.optim as optim


import time 
import json 
import fastText

import numpy as np
import time
import random
import nltk


class Trainer:
    def __init__(self, input_dir, model, batch_size, lr, epoch = 10, class_num = 3):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.input_dir = input_dir 
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.class_num = class_num
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.criterion = nn.CrossEntropyLoss()
    
        self.caption_train = json.load(open(self.input_dir+"train_caption_json.json", 'r'))[:-300]
        self.caption_val = json.load(open(self.input_dir+"train_caption_json.json", 'r'))[-300:]
        self.reference_train = json.load(open(self.input_dir+"train_reference_json.json", 'r'))[:-300]
        self.reference_val = json.load(open(self.input_dir+"train_reference_json.json", 'r'))[-300:]
        self.negative_train = json.load(open(self.input_dir+"train_negative_json.json", 'r'))[:-300]
        self.negative_val = json.load(open(self.input_dir+"train_negative_json.json", 'r'))[-300:]

        # self.word2vec_model = word2vec.load('./embedding/word2vec.bin')
        self.word2vec_model = fastText.FastText.load_model('./embedding/fil9.bin')

    def train(self):
        min_len_train = np.min(np.array([len(self.caption_train), len(self.reference_train), len(self.negative_train)]))
        # print(min_len_train, min_len_val)

        category_batch_num = int(self.batch_size/self.class_num)
        idx = int(min_len_train // category_batch_num)

        global_step = 0
        start_time = time.time()
        for ep in range(self.epoch):
            self.model.train()

            if ep == int(self.epoch //3):
                self.lr = self.lr/10
                self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
            if ep == int(self.epoch*2//3):
                self.lr = self.lr/10
                self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)

            for idi in range(idx):

                batch_negative_list = random.sample(self.negative_train, category_batch_num)
                batch_negative = sentence_embedding(batch_negative_list, self.word2vec_model, 0)

                batch_reference_list = random.sample(self.reference_train, category_batch_num)
                batch_reference = sentence_embedding(batch_reference_list, self.word2vec_model, 1)

                batch_caption_list = random.sample(self.caption_train, category_batch_num)
                batch_caption = sentence_embedding(batch_caption_list, self.word2vec_model, 2)

                batch_data_n = batch_negative + batch_reference + batch_caption
                random.shuffle(batch_data_n)
                batch_data = []
                batch_label = []
                for i in range(len(batch_data_n)):
                    batch_data.append(batch_data_n[i][0][0])
                    batch_label.append(batch_data_n[i][0][1])

                batch_data = torch.FloatTensor(batch_data).to(self.device)
                batch_label = torch.LongTensor(batch_label).to(self.device)
                
                self.optimizer.zero_grad()

                output_logits = self.model(batch_data)
                try:
                    if self.model.aux_logits:
                        _loss = [] 
                        for i in range(len(output_logits)):
                            _loss.append(self.criterion(output_logits[i], batch_label))
                        loss = sum(_loss)
                        preds = torch.argmax(output_logits[0], dim = -1)
                except:
                    loss = self.criterion(output_logits, batch_label)
                    preds = torch.argmax(output_logits, dim = -1)

                # preds_matrix = nn.Softmax()(output_logits)
                # preds = torch.argmax(output_logits, dim = -1)

                acc = torch.sum(preds == batch_label).float()/self.batch_size

                loss.backward()
                self.optimizer.step()
                print("Epoch:[{}]===Step:[{}/{}]===Time:[{:.2f}]===Learning Rate:{}\nTrain_loss:[{:.4f}], Train_acc[{:.4f}]".format(ep, idi, idx, time.time()-start_time, self.lr, loss, acc))
                
                global_step += 1

            self.validation(global_step)

    def validation(self, global_step):
        min_len_val = np.min(np.array([len(self.caption_val), len(self.reference_val), len(self.negative_val)]))
        val_catagory_batch = int(self.batch_size/self.class_num)
        val_idx = int(min_len_val // val_catagory_batch)

        val_loss_sum = 0
        val_acc_sum = 0

        self.model.eval()

        for val_idi in range(val_idx):
            print(val_idi)
            val_batch_negative_list = random.sample(self.negative_train, val_catagory_batch)
            val_batch_negative = sentence_embedding(val_batch_negative_list, self.word2vec_model, 0)

            val_batch_reference_list = random.sample(self.reference_train, val_catagory_batch)
            val_batch_reference = sentence_embedding(val_batch_reference_list, self.word2vec_model,1)

            val_batch_caption_list = random.sample(self.caption_train, val_catagory_batch)
            val_batch_caption = sentence_embedding(val_batch_caption_list, self.word2vec_model,2)

            val_batch_data_n = val_batch_negative + val_batch_reference + val_batch_caption
            random.shuffle(val_batch_data_n)
            val_batch_data = []
            val_batch_label = []
            for i in range(len(val_batch_data_n)):
                val_batch_data.append(val_batch_data_n[i][0][0])
                val_batch_label.append(val_batch_data_n[i][0][1])
            

            val_batch_data = torch.FloatTensor(val_batch_data).to(self.device)
            val_batch_label = torch.LongTensor(val_batch_label).to(self.device)

            output_logits = self.model(val_batch_data)

            try:
                if self.model.aux_logits:
                    loss = self.criterion(output_logits[0], val_batch_label)
                    preds = torch.argmax(output_logits[0], dim = -1)
            except:
                loss = self.criterion(output_logits, val_batch_label)
                preds = torch.argmax(output_logits, dim = -1)

            acc = torch.sum(preds == val_batch_label).float()/self.batch_size

            print(loss)

            val_loss_sum += loss
            val_acc_sum += acc

        print("\n===\nValidation Loss: {:.4f}, Validation Acc: {:.4f}\n===\n".format(val_loss_sum/val_idx, val_acc_sum/val_idx))

class Predictor:
    def __init__(self, p_input, model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.p_input = p_input 
        self.model = model.to(self.device)
        self.model.eval()
        self.word2vec_model = fastText.FastText.load_model('./embedding/fil9.bin')
    def predict(self):
        embedded_input = prediction_embedding(self.p_input, self.word2vec_model)
        print(np.array(embedded_input).shape)
        embedded_input = torch.FloatTensor(embedded_input).to(self.device)
        logits = self.model(embedded_input)
        results = F.softmax(logits, dim=1)
        clss = torch.argmax(results, dim=1)
        if clss == 0:
            print("Normal Text")
        elif clss == 1:
            print("Reference")
        else:
            print("Caption")


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
        sent_matrix = np.expand_dims(np.array(sent_matrix), axis = 0)
        pos_matrix = np.expand_dims(np.array(pos_matrix), axis = 0)
        concated_matrix = np.concatenate((sent_matrix, pos_matrix), axis = 0).tolist()

        sample.append([concated_matrix, category])

        batch_list.append(sample)
    
    return batch_list

def prediction_embedding(p_input, word2vec_model):

    batch_list = []
    if isinstance(p_input, str):
        token_list = nltk.word_tokenize(p_input)

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
        sent_matrix = np.expand_dims(np.array(sent_matrix), axis = 0)
        pos_matrix = np.expand_dims(np.array(pos_matrix), axis = 0)
        concated_matrix = np.concatenate((sent_matrix, pos_matrix), axis = 0).tolist()

        batch_list.append(concated_matrix)

    elif isinstance(p_input, list):
        for sentence in p_input:
            token_list = nltk.word_tokenize(sentence)

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
            sent_matrix = np.expand_dims(np.array(sent_matrix), axis = 0)
            pos_matrix = np.expand_dims(np.array(pos_matrix), axis = 0)
            concated_matrix = np.concatenate((sent_matrix, pos_matrix), axis = 0).tolist()

            batch_list.append(concated_matrix)

    else:
        print("Input should be a sentence string or a list of sentence string")
        raise TypeError
            
    return batch_list