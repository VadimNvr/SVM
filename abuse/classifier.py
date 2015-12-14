from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_union

import json
import re


# Takes all words
def word_tokenizer(text):
    return [tok.strip().lower() for tok in re.findall(r'\b\w+\b', re.sub(r'\d', ' ', text)) if len(tok) > 1]


# Takes all pairs of words, one of which starts with "you"
def pair_tokenizer(text):
    words = word_tokenizer(text)
    return [i+j for i in words for j in words if i >= "you" and i != j]


class Classifier:

    def __init__(self):
        word_vectoriser = TfidfVectorizer(tokenizer=word_tokenizer)
        pair_vectoriser = TfidfVectorizer(tokenizer=pair_tokenizer)

        feature_extractor = make_union(word_vectoriser, pair_vectoriser)
        self.pipeline = make_pipeline(feature_extractor, LinearSVC(C=0.25))

    def fit(self, data, labels):
        self.pipeline.fit(data, labels)

    def fit_transform(self, data, labels):
        return self.pipeline.fit_transform(data, labels)

    def predict(self, data):
        return self.pipeline.predict(data)


if __name__ == "__main__":

    train_data = []
    train_lbls = []

    for line in open("./data/data_set.json", "r"):
        data = json.loads(line)
        train_data.append(data["data"])
        train_lbls.append(data["label"])

