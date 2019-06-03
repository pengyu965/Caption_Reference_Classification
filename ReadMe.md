# Requirement


- word2vec

```python
pip3 install word2vec
```

- nltk

```python
pip3 install -U nltk
```

- tensorflow 1.13.0

```python
pip3 install tensorflow-gpu==1.13.0 keras
```

- numpy 

```python
pip3 install numpy
```

- others

```python
pip3 install re json html lxml
```

# Preprocess

### Run preprocess_1.py first:

```python
python3 ./preprocess/preprocess_1.py
```

- Getting ann files from SCISUMM-DATA, stored in './original_data/annfile/'

- Getting xml files from SCISUMM-DATA, stored in './original_data/xmlfile/'

- Convert the ann file into JSON, stored in './cache_data/ann_jsonfile/' (ann2json.py is used in this step)

(xml file is used for later negative samle generation)

### Secondly run negative_sample_gen.py

```python
python3 ./preprocess/negative_sample_gen.py
```


- The xml file which doesn't contain any annotated sentence would be stored in './cache_data/negative_xmlfile/' (According to ann_json file to remove the annotated sentence)

- The txt files, which are obtained from removing xml tags from xml files, are stored in './cache_data/negative_txtfile/'

- Then convert these negative samples into JSON version, which is stored in './cache_data/neg_jsonfile/'

### Thirdly run preprocess_2.py

```python
python3 ./preprocess/preprocess_2.py
```

- All caption sentences would be stored in "train_caption_json.json" in "./traindata/" folder

- All reference sentences would be stored in "train_reference_json.json" in "./traindata/" folder

- All normal sentences would be stored in "train_negative_json.json" in "./traindata/" folder

- **Only** maintain the sentence which the token length between **[6, 200]**

### Word2Vec

```python
python3 ./embedding/word2vec_train.py
```

- word2vec_train.py is used to add all the word tokens in the training data to the vocabulary, which is './embedding/vocabulary.txt'

- word2vec_train.py is used to train the word2vec model and store the model in './embedding/word2vec.bin'


# CNN and LSTM model

### CNN model training

```python
python3 main.py --model=CNN --train --data=./traindata/ --lr=0.001 --epoch=30 --bsize=60 
```

### LSTM model training

```python
python3 main.py --model=LSTM --train --data=./traindata/ --lr=0.001 --epoch=60 --bsize=60 
```