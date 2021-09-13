# coding=utf-8
import os

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from sklearn.feature_extraction.text import CountVectorizer

sw = stopwords.words('russian')
m = Mystem()
docs = []
for root, dirs, files in os.walk('friends-data'):
    for name in files:
        filepath = os.path.join(root, name)
        with open(filepath, 'r') as f:
            docs.append(f.read())


def preprocess_text(text):
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    words = [w for w in words if w not in sw]
    text = m.lemmatize(' '.join(words))
    return ''.join(text)[:-1]


def index_dict(texts):
    corpus = [preprocess_text(i) for i in texts]
    vectorizer = CountVectorizer(analyzer='word')
    count_mtx = vectorizer.fit_transform(corpus).toarray()
    mtx_freq = np.asarray(count_mtx.sum(axis=0)).ravel()
    frq = mtx_freq.astype(int)
    ind_dict = {}
    for i, word in enumerate(vectorizer.get_feature_names()):
        ind_dict[word] = {'docs': np.nonzero(count_mtx[:, i])[0], 'freq': frq[i]}
    return ind_dict


ind_d = index_dict(docs)
mon = ('Моника', ind_d['моника']['freq'] + ind_d['мон']['freq'])
chen = ('Чендлер', ind_d['чендлер']['freq'] + ind_d['чен']['freq'] + ind_d['чэндлер']['freq'])
rach = ('Рейчел', ind_d['рэйчел']['freq'] + ind_d['рейч']['freq'] + ind_d['рейчел']['freq'])
ross = ('Росс', ind_d['росс']['freq'])
joe = ('Джоуи', ind_d['джоуи']['freq'] + ind_d['джо']['freq'])
phib = ('Фиби', ind_d['фиби']['freq'] + ind_d['фибс']['freq'])
heros = [mon, chen, rach, ross, joe, phib]

print('Самый популярный герой:', max(heros, key=lambda x: x[1])[0])
print('Самое популярное слово:', max(ind_d.items(), key=lambda x: x[1]['freq'])[0])
print('Самое редкое слово:', min(ind_d.items(), key=lambda x: x[1]['freq'])[0])

wds = []
for word in ind_d.keys():
    if len(ind_d[word]['docs']) == len(docs):
        wds.append(word)
print('Слова, которые есть во всех документах:', ', '.join(wds))
