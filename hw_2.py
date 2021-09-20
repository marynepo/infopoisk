import os
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.spatial.distance as distance

sw = stopwords.words('russian')
m = Mystem()


def make_corpus(path):
    docs = []
    docnames = []
    for root, dirs, files in os.walk(path):
        for name in files:
            filepath = os.path.join(root, name)
            with open(filepath, 'r') as f:
                docs.append(f.read())
                docnames.append(name)
    return docs, docnames


def preprocess_text(text):
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    words = [w for w in words if w not in sw]
    text = m.lemmatize(' '.join(words))
    return ''.join(text)[:-1]


def index_mtx(docs):
    corpus = [preprocess_text(d) for d in docs]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    return X, vectorizer


def ind_query(qr, vct):
    qr = preprocess_text(qr)
    return vct.transform([qr]).toarray()


def calc_distance(qr, ind_mtx):
    dist = 1 - distance.cdist(ind_mtx, qr, 'cosine')
    return dist


def main(qr):
    docs, docnames = make_corpus('friends-data')
    ind_mtx, vct = index_mtx(docs)
    ind_q = ind_query(qr, vct)
    sort_ind = calc_distance(ind_q, ind_mtx).argsort(axis=0)[-1::-1]
    return [docnames[i[0]] for i in sort_ind]


# Пример. Запрос - начало 1 серии 1 сезона
print(', '.join(main('''Да нечего рассказывать!
Он просто сотрудник!

Ладно тебе, ты же на свидание
с ним собралась!

Значит, он не может не быть с придурью!

Джои, веди себя прилично

Так у него горб? И парик в придачу?

Погодите, я знаю – он ест мел?

Ну, знаете, я просто не хочу, чтобы она
пережила то же, что я с Карлом… Ой!

Ладно вам, успокойтесь.

Успокойтесь. Это даже не свидание.
''')))
