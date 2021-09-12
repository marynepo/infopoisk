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


def index_mtx(texts):
    corpus = [preprocess_text(i) for i in texts]
    vectorizer = CountVectorizer(analyzer='word')
    ind_mtx = vectorizer.fit_transform(corpus)
    matrix_freq = np.asarray(ind_mtx.sum(axis=0)).ravel()
    return ind_mtx, np.array([np.array(vectorizer.get_feature_names()), matrix_freq])


X, mtx = index_mtx(docs)
frq = mtx[1, :].astype(float)
mon = ('Моника', frq[np.where(mtx[0] == 'моника')][0] + frq[np.where(mtx[0] == 'мон')][0])
chen = ('Чендлер',
        frq[np.where(mtx[0] == 'чендлер')][0] + frq[np.where(mtx[0] == 'чен')][0] + frq[np.where(mtx[0] == 'чэндлер')][
            0])
rach = ('Рейчел',
        frq[np.where(mtx[0] == 'рэйчел')][0] + frq[np.where(mtx[0] == 'рейч')][0] + frq[np.where(mtx[0] == 'рейчел')][
            0])
ross = ('Росс', frq[np.where(mtx[0] == 'росс')][0])
joe = ('Джоуи', frq[np.where(mtx[0] == 'джоуи')][0] + frq[np.where(mtx[0] == 'джо')][0])
phib = ('Фиби', frq[np.where(mtx[0] == 'фиби')][0] + frq[np.where(mtx[0] == 'фибс')][0])
heros = [mon, chen, rach, ross, joe, phib]

print('Самый популярный герой:', max(heros, key=lambda x: x[1])[0])
print('Самое частотное слово:', mtx[0][np.argmax(frq)])
print('Самое редкое слово:', mtx[0][np.argmin(frq)])
print(
    'Слова, которые есть во всех документах:',
    ', '.join(mtx[0][np.where(np.prod(X.toarray().astype(float), axis=0) != 0)]))
