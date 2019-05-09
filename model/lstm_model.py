import tensorflow as tf
import numpy as np 
import os
import argparse
import scipy.misc
import time
import keras
import json
import random
import nltk
import word2vec
import fastText


class Model():
    def __init__(self, batch_size=50, lr = 0.001, keep_prob=0.4, class_num=3, is_training = True):
        self.layer = {}
        self.batch_size = batch_size
        self.lr = lr 
        self.keep_prob = keep_prob
        self.class_num = class_num
        self.is_training = is_training

        self.input = tf.placeholder(tf.float32, [None, None, 100])
        self.label = tf.placeholder(tf.int64, [None])

    def lstm_layer(self):
        with tf.name_scope("stacked_LSTMCell"):
            cells = [
                tf.keras.layers.LSTMCell(128),
                tf.keras.layers.LSTMCell(128),
                tf.keras.layers.LSTMCell(128),
            ]
        with tf.name_scope("LSTM_layer"):
            self.layer["lstm_output"]= tf.keras.layers.RNN(cells, input_shape = [None, None, 100])(self.input)
        
    def fc(self):
        with tf.variable_scope('fc1',reuse=tf.AUTO_REUSE):
            self.layer['fc1_layer'] = tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu)(self.layer['lstm_output'])
        
        with tf.variable_scope('fc_output', reuse=tf.AUTO_REUSE):
            self.layer['logits'] = tf.keras.layers.Dense(units = self.class_num)(self.layer['fc1_layer'])
        
        with tf.variable_scope('softmax'):
            self.layer['softmax'] = tf.keras.activations.softmax(self.layer['logits'])

    def loss(self):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label,
                logits=self.layer['logits'],
                name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy)
        # tf.summary.scalar("train_loss", loss)
        return loss

    def accuracy(self):
        with tf.name_scope('accuracy'):
            prediction = tf.argmax(self.layer['logits'], axis=1)
            correct_prediction = tf.equal(prediction, self.label)
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), 
            name = 'result')
        # tf.summary.scalar("train_acc", accuracy)
        return accuracy

    def optimizer(self):
        return tf.train.AdamOptimizer(self.lr)


class Trainer:
    def __init__(self, input_dir, model, epoch = 10, class_num = 3):
        self.input_dir = input_dir 
        self.model = model 
        self.epoch = epoch
        self.class_num = class_num

        self.lr = self.model.lr
        self.batch_size = self.model.batch_size
        
        self.model.lstm_layer()
        self.model.fc()
        self.loss = self.model.loss()
        self.accuracy = self.model.accuracy()
        self.optimizer = self.model.optimizer()

        self.train_op = self.optimizer.minimize(self.loss)

        self.caption_train = json.load(open(self.input_dir+"train_caption_json.json", 'r'))[:-300]
        self.caption_val = json.load(open(self.input_dir+"train_caption_json.json", 'r'))[-300:]
        self.reference_train = json.load(open(self.input_dir+"train_reference_json.json", 'r'))[:-300]
        self.reference_val = json.load(open(self.input_dir+"train_reference_json.json", 'r'))[-300:]
        self.negative_train = json.load(open(self.input_dir+"train_negative_json.json", 'r'))[:-300]
        self.negative_val = json.load(open(self.input_dir+"train_negative_json.json", 'r'))[-300:]

        # self.word2vec_model = word2vec.load('./embedding/word2vec.bin')
        self.word2vec_model = fastText.FastText.load_model('./embedding/fil9.bin')
        

    def train(self, sess, writer):

        min_len_train = np.min(np.array([len(self.caption_train), len(self.reference_train), len(self.negative_train)]))
        # print(min_len_train, min_len_val)

        category_batch_num = int(self.batch_size/self.class_num)
        idx = int(min_len_train // category_batch_num)

        global_step = 0
        start_time = time.time()
        for ep in range(self.epoch):
            if ep == int(self.epoch //3):
                self.lr = self.lr / 10
            if ep == int(self.epoch*2//3):
                self.lr = self.lr / 10

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
                # print(np.array(batch_data).shape, np.array(batch_label).shape)
                # batch_label = [0]*category_batch_num +[1]*category_batch_num+[2]*category_batch_num

                # print(batch_data_n)
                _, loss_val, acc_val= sess.run(
                    (self.train_op, self.loss, self.accuracy),
                    feed_dict={self.model.input: np.array(batch_data)[:,:50,:], self.model.label: batch_label}
                )

                print("Epoch:[{}]===Step:[{}/{}]===Time:[{:.2f}]===Learning Rate:{}\nTrain_loss:[{:.4f}], Train_acc[{:.4f}]"
                    .format(ep, idi, idx, time.time()-start_time, self.lr, loss_val, acc_val))
                # print(eee)

                manual_summary = tf.Summary(
                    value = [
                        tf.Summary.Value(tag='train_acc', simple_value = acc_val * 1.), 
                        tf.Summary.Value(tag='train_loss', simple_value = loss_val * 1.)
                        ]
                )
                writer.add_summary(manual_summary, global_step)


                global_step += 1

            self.validation(sess, writer, global_step)

    def validation(self, sess, writer, global_step):
        min_len_val = np.min(np.array([len(self.caption_val), len(self.reference_val), len(self.negative_val)]))
        val_catagory_batch = int(self.batch_size/self.class_num)
        val_idx = int(min_len_val // val_catagory_batch)

        val_loss_sum = 0
        val_acc_sum = 0

        for val_idi in range(val_idx):
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
            

            val_loss_val, val_acc_val= sess.run(
                    (self.loss, self.accuracy),
                    feed_dict={self.model.input: np.array(val_batch_data)[:,:50,:], self.model.label: val_batch_label}
                )
            val_loss_sum += val_loss_val
            val_acc_sum += val_acc_val

        print("\n===\nValidation Loss: {:.4f}, Validation Acc: {:.4f}\n===\n".format(val_loss_sum/val_idx, val_acc_sum/val_idx))

        val_manual_summary = tf.Summary(
                value = [
                    tf.Summary.Value(tag='val_acc', simple_value = val_acc_sum * 1. /val_idx), 
                    tf.Summary.Value(tag='val_loss', simple_value = val_loss_sum * 1. /val_idx)
                    ]
            )
        writer.add_summary(val_manual_summary, global_step)
            

def sentence_embedding(dic_list, word2vec_model, category):
    batch_list = []
    for dic in dic_list:
        token_list = nltk.word_tokenize(dic["Text"])
        
        sample = []

        sent_matrix = []

        for token in token_list:
            # word_vec = word2vec_model[token].tolist()
            word_vec = word2vec_model.get_word_vector(token).tolist()
            
            sent_matrix.append(word_vec)
        
        sent_matrix += [[0]*100]*(200-len(sent_matrix))

        sample.append([sent_matrix, category])

        batch_list.append(sample)
    
    return batch_list

