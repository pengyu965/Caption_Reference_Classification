import word2vec
from string import punctuation
import os
import re
import json 
import nltk
import numpy as np

def vocabulary_gen(caption_data, reference_data, negative_data):
    with open(caption_data, 'r') as f:
        caption_j = json.load(f)
    with open(reference_data, 'r') as f:
        reference_j = json.load(f)
    with open(negative_data, 'r') as f:
        negative_j = json.load(f)

    j_list = [caption_j, reference_j, negative_j]

    length = []

    for j_file in j_list:
        for i in range(len(j_file)):
            word_list = nltk.word_tokenize(j_file[i]["Text"])
            # print(word_list)
            length.append(len(word_list))
            voca_writer(word_list)
    
    max_length = np.max(np.array(length))
    min_length = np.min(np.array(length))
    print("The longest sentence is:", max_length)
    print("The shortest sentence is:", min_length)


def voca_writer(word_list):
    for word in word_list:
        with open("./embedding/vocabulary.txt", 'a') as f:
            f.write(word+' ')




if __name__ == "__main__":
    caption_data = "./traindata/train_caption_json.json"
    reference_data = "./traindata/train_reference_json.json"
    negative_data = "./traindata/train_negative_json.json"
    vocabulary_gen(caption_data, reference_data, negative_data)

    word2vec.word2vec('./embedding/vocabulary.txt', './embedding/word2vec.bin', size=100, min_count=0, verbose=True)

    model = word2vec.load('./embedding/word2vec.bin')
    