from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_union
import itertools
from sklearn.metrics import f1_score
from sklearn import cross_validation
from numpy.ma import mean

import json
import re


#Выбирает из входного текста слова (букв хотя бы больше одной)
def word_tokenizer(text):
    return [tok.strip().lower() for tok in re.findall(r'\b\w+\b', re.sub(r'\d', ' ', text)) if len(tok) > 1]


#Выбирает из входного текста сочетания из 2 букв
def ending_tokenizer(text):
    return [tok.strip().lower() for tok in re.findall(r'\w\w', re.sub(r'\d', ' ', text))]


#Выбирает из входного текста все буквы
def char_tokenizer(text):
    return [tok.strip().lower() for tok in re.findall(r'\w', re.sub(r'\d', ' ', text))]


train_data = []
train_lbls = []

for line in open("./data/data_set.json", "r"):
    data = json.loads(line)
    train_data.append(data["data"])
    train_lbls.append(int(data["label"] == "EN")) # Класс 1 - Английский, класс 0 - Тагальский

#Создаём 3 извлекателя признаков (по словам, по парам букв, по буквам)
word_vectoriser = TfidfVectorizer(tokenizer=word_tokenizer)
ends_vectoriser = TfidfVectorizer(tokenizer=ending_tokenizer)
char_vectorizer = TfidfVectorizer(tokenizer=char_tokenizer)

#Группируем наши feature extractor-ы и создаём конвейер
feature_extractor = make_union(word_vectoriser, ends_vectoriser, char_vectorizer)
pipeline = make_pipeline(feature_extractor, LinearSVC(C=2))


#pipeline.fit(train_data[::2], train_lbls[::2])
#print(f1_score(train_lbls[1::2], pipeline.predict(train_data[1::2])))

scores = cross_validation.cross_val_score(pipeline, train_data, train_lbls, cv=5, scoring='f1_macro')
print(mean(scores))




