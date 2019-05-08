import os
import ann2json
import shutil
import json 
import random
import nltk
import matplotlib.pyplot as plt

def train_caption_gen(ann_json_path, train_data_path):
    train_json = []
    for j_file in os.listdir(ann_json_path):
        with open(os.path.join(ann_json_path, j_file)) as f:
            js = json.load(f)
        
        for i in range(len(js)):
            # if js[i]["Entity"] == "Caption" and len(nltk.word_tokenize(js[i]["Text"])) > 5 and len(nltk.word_tokenize(js[i]["Text"])) <= 200:
            dic = {}
            dic["Text"] = js[i]["Text"]
            dic["Entity"] = js[i]["Entity"]
            train_json.append(dic)

    ## Adding some captions from figure&extraction result
    with open("./truepaired_data.json", 'r') as f:
        fe_c = json.load(f)

    sampled_fe_c = random.sample(fe_c, 1500)

    for i in range(len(sampled_fe_c)):
        # if len(nltk.word_tokenize(sampled_fe_c[i]["Caption"])) > 5 and len(nltk.word_tokenize(sampled_fe_c[i]["Caption"])) <= 200:
        dic = {}
        dic["Text"] = sampled_fe_c[i]["Caption"]
        dic["Entity"] = "Caption"
        train_json.append(dic)

    shuffle_train_json = random.sample(train_json, len(train_json))
    print("Caption sample number:", len(shuffle_train_json))
    with open(train_data_path+"/train_caption_json.json", 'w') as f:
        f.write(json.dumps(shuffle_train_json, indent=4))

def train_reference_gen(ann_json_path, train_data_path):
    train_json = []
    for j_file in os.listdir(ann_json_path):
        with open(os.path.join(ann_json_path, j_file)) as f:
            js = json.load(f)
        
        for i in range(len(js)):
            # if js[i]["Entity"] == "Reference" and len(nltk.word_tokenize(js[i]["Text"])) > 5 and len(nltk.word_tokenize(js[i]["Text"])) <= 200:
            dic = {}
            dic["Text"] = js[i]["Text"]
            dic["Entity"] = js[i]["Entity"]
            train_json.append(dic)
            train_json.append(dic)
            train_json.append(dic)

    shuffle_train_json = random.sample(train_json, len(train_json))

    print("Reference sample number:", len(shuffle_train_json))
    with open(train_data_path+"/train_reference_json.json", 'w') as f:
        f.write(json.dumps(shuffle_train_json, indent=4))


def train_negative_gen(neg_json_path, train_data_path):
    train_json = []

    for j_file in os.listdir(neg_json_path):
        with open(os.path.join(neg_json_path, j_file)) as f:
            neg_j = json.load(f)
            
        for i in range(len(neg_j)):
            # if len(nltk.word_tokenize(neg_j[i]["Text"]))>5 and len(nltk.word_tokenize(neg_j[i]["Text"])) <= 200:
            train_json.append(neg_j[i])

    print("negative sample number:",len(train_json))
    with open(train_data_path+"/train_negative_json.json", 'w') as f:
        f.write(json.dumps(train_json, indent=4))

def histogram_gen(train_data_path):
    for file in os.listdir(train_data_path):
        print(file)
        file_name, _ = os.path.splitext(file)
        file_path = os.path.join(train_data_path,file)
        length = []
        with open(file_path,'r') as jsonfile:
            json_l = json.load(jsonfile)
        for i in range(len(json_l)):
            text_length = len(nltk.word_tokenize(json_l[i]["Text"]))
            length.append(text_length)

        drawer(length, train_data_path+file_name)

def drawer(a_list, save_name):
    # plt.figure(figsize=(16.00,12.00), dpi=100)
    plt.hist(a_list, bins=200, ec="black")
    plt.xlabel('Sentence token length')
    plt.ylabel('Number')
    plt.show()
    # plt.savefig(save_name)
    # plt.close()






if __name__ == "__main__":
    ann_json_path = "./ann_jsonfile/"
    neg_json_path = "./neg_jsonfile/"
    train_data_path = "./nofiltered_traindata/"

    if os.path.exists(train_data_path) == False:
        os.mkdir(train_data_path)

    # train_caption_gen(ann_json_path, train_data_path)
    # train_reference_gen(ann_json_path, train_data_path)
    # train_negative_gen(neg_json_path, train_data_path)
    histogram_gen(train_data_path)

