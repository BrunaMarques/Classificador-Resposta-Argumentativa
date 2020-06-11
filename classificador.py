# -*- coding: utf-8 -*-

import json
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def to_sentences(abstracts, senteces_max=None):
    sentences = []
    labels = []
    abstracts_sentences = []
    abstracts_labels = []

    for id, abstract in enumerate(abstracts):
        if senteces_max and len(abstract) > senteces_max:
            continue

        tmp_sentences = []
        tmp_labels = []

        for label, text in abstract:
            sentences.append(text)
            labels.append(label)

            tmp_sentences.append(text)
            tmp_labels.append(label)

        abstracts_sentences.append(tmp_sentences)
        abstracts_labels.append(tmp_labels)

    assert (len(sentences) == len(labels))
    assert (len(abstracts_sentences) == len(abstracts_labels))

    return sentences, labels, abstracts_sentences, abstracts_labels


def loadJson(file):
    data = []
    with open(file, encoding='utf-8') as f:
        # data = json.load(f, encoding='cp1252')
        data = json.load(f)

    return to_sentences(data)


def abstracts_to_sentences(abstracts, labels):
    ret = []
    ret_prev = []
    ret_next = []
    ret_labels = []
    ret_pos = []

    for i, (sentences_labels, sentences) in enumerate(zip(labels, abstracts)):
        for j, (label, sentence) in enumerate(zip(sentences_labels, sentences)):
            ret.append(sentence)
            ret_pos.append(j)
            ret_labels.append(label)

            if j - 1 >= 0:
                ret_prev.append(sentences[j - 1])
            else:
                ret_prev.append("")

            if j + 1 < len(sentences):
                ret_next.append(sentences[j + 1])
            else:
                ret_next.append("")

    return ret, ret_prev, ret_next, ret_pos, ret_labels


def classificador():

    corpus = 'padrao_ouroOK.json'

    ngrama = 1

    k = 100

    print("lendo arquivo")
    _, _, data, labels, _ = loadJson(corpus)
    X_sentences, X_prev, X_next, X_pos, Y_sentences, _ = abstracts_to_sentences(
        data, labels)

    print("tfidf")
    vectorizer = TfidfVectorizer(ngram_range=(1, ngrama))
    X_sentences = vectorizer.fit_transform(X_sentences)
    X_prev = vectorizer.transform(X_prev)
    X_next = vectorizer.transform(X_next)

    print(len(vectorizer.get_feature_names()))

    print("chi-quadrado")
    selector = SelectKBest(chi2, k=k)
    X_sentences = selector.fit_transform(X_sentences, Y_sentences)
    X_prev = selector.transform(X_prev)
    X_next = selector.transform(X_next)

    print("adicionando anterior e posterior")
    X_sentences = hstack([X_sentences, X_prev, X_next,
                          np.expand_dims(np.array(X_pos), -1)])

    print("Inicializando classificador...")
    clf = LinearSVC(dual=False, tol=1e-3)
    #clf =  LogisticRegression(random_state=0)

    print("Predicão...")
    pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=10)

    print("Classification_report:")
    print(classification_report(Y_sentences, pred))
    print("")
    print(confusion_matrix(Y_sentences, pred))


classificador()
