import json
import nltk
from scipy import sparse
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# nltk.download('stopwords')
sw = stopwords.words('russian')
m = Mystem()


def make_corpus(path):
    with open(path, 'r') as f:
        corpus = list(f)[:50000]
    answers = []
    for doc in corpus:
        doc = json.loads(doc)
        ans = sorted(doc['answers'], key=lambda x: x['author_rating']['value'])
        if len(ans) > 0:
            answers.append(ans[-1]['text'])
    return answers


def preprocess_text(text):
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    words = [w for w in words if w not in sw]
    text = m.lemmatize(' '.join(words))
    return ''.join(text)[:-1]


def index_mtx(docs):
    texts = [preprocess_text(d) for d in docs]
    count_vectorizer = CountVectorizer()
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    x_count_vec = count_vectorizer.fit_transform(texts)
    x_tf_vec = tf_vectorizer.fit_transform(texts)
    x_tfidf_vec = tfidf_vectorizer.fit_transform(texts)
    idf = tfidf_vectorizer.idf_
    tf = x_tf_vec
    k = 2
    b = 0.75
    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()
    B_1 = (k * (1 - b + b * len_d/avdl))
    values = []
    rows = []
    cols = []
    for i, rc in enumerate(zip(*tf.nonzero())):
        val_a = tf.data[i] * idf[rc[1]] * (k+1)
        val_b = tf.data[i] + float(B_1[rc[0]])
        values.append(val_a/val_b)
        rows.append(rc[0])
        cols.append(rc[1])
    mtx = sparse.csr_matrix((values, (rows, cols)))
    return mtx, count_vectorizer


def ind_query(qr, vct):
    return vct.transform([qr])


def calc_distance(ind_qr, ind_mtx):
    return np.dot(ind_mtx, ind_qr.T).toarray()


def main():
    answers = make_corpus('data.jsonl')
    # у меня не хватает памяти для хранения answers в арэе, поэтому каждому элементу соответствует айди (чтобы мотом можно было реализовать сортировку через маску)
    ans_ids = np.arange(0, len(answers)*3, 3)
    ind_mtx, vct = index_mtx(answers)
    qr = input('Введите запрос или нажмите на Enter, если больше не хотите спрашивать: ')
    results = []
    while qr != '':
        qr = preprocess_text(qr)
        ind_q = ind_query(qr, vct)
        sort_ind = calc_distance(ind_q, ind_mtx).argsort(axis=0)[-1::-1]
        res = ans_ids[sort_ind.ravel()]
        results.append(res)
        print('Топ 5:\n', '\n'.join([answers[i//3] for i in res[:5]]))
        qr = input('Введите запрос или нажмите на Enter, если больше не хотите спрашивать: ')
    return results


if __name__ == '__main__':
   results = main()
